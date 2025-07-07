# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import librosa
import noisereduce as nr
import scipy.signal
import pywt
import os
import uuid
from tempfile import NamedTemporaryFile
import soundfile as sf
from pydub import AudioSegment
import io

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def convert_to_wav(audio_file):
    """Convert any audio file to WAV format using pydub"""
    try:
        # Read the audio file
        audio = AudioSegment.from_file(io.BytesIO(audio_file))
        # Convert to WAV
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io.read()
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {str(e)}")

def spectral_gating(audio, sample_rate):
    return nr.reduce_noise(y=audio, sr=sample_rate, stationary=False)

def wavelet_denoising(audio):
    coeffs = pywt.wavedec(audio, 'db8', level=6)
    threshold = np.median(np.abs(coeffs[-1])) / 0.675
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, 'db8')

def adaptive_threshold(audio):
    threshold = np.mean(np.abs(audio)) * 0.5
    return np.where(np.abs(audio) > threshold, audio, 0)

def bandpass_filter(audio, lowcut=300, highcut=3400, sample_rate=44100, order=6):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.lfilter(b, a, audio)

def apply_advanced_noise_reduction(audio, sample_rate):
    step1 = spectral_gating(audio, sample_rate)
    step2 = wavelet_denoising(step1)
    step3 = adaptive_threshold(step2)
    step4 = bandpass_filter(step3)
    return step4

@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        file_contents = await file.read()
        
        # First try to convert to WAV
        try:
            wav_data = convert_to_wav(file_contents)
            temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4().hex}.wav")
            with open(temp_path, "wb") as f:
                f.write(wav_data)
        except Exception as conv_e:
            # If conversion fails, try to load directly
            temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4().hex}")
            with open(temp_path, "wb") as f:
                f.write(file_contents)
        
        # Load audio file with multiple fallbacks
        try:
            audio, sample_rate = librosa.load(temp_path, sr=None)
        except Exception as load_error:
            # Try with soundfile as fallback
            try:
                audio, sample_rate = sf.read(temp_path)
                if audio.ndim > 1:  # Convert stereo to mono if needed
                    audio = np.mean(audio, axis=1)
            except Exception as sf_error:
                raise ValueError(
                    f"Failed to load audio file. Librosa error: {str(load_error)}. "
                    f"SoundFile error: {str(sf_error)}"
                )
        
        # Apply noise reduction
        cleaned_audio = apply_advanced_noise_reduction(audio, sample_rate)
        
        # Save the cleaned audio
        output_filename = f"cleaned_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        sf.write(output_path, cleaned_audio, sample_rate)
        
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return {"filename": output_filename}
    
    except Exception as e:
        # Clean up temporary files if they exist
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)