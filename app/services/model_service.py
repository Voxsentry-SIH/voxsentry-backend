import os
import logging
import numpy as np
import librosa
import tensorflow as tf
from app.config import MODEL_PATH

logger = logging.getLogger(__name__)

# Load model at import time
model = None
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file not found at {MODEL_PATH}. Prediction endpoints will fail.")
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Loaded voice clone detector model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {str(e)}")

def preprocess_audio(audio_path: str) -> np.ndarray:
    """
    Preprocess audio exactly matching the training notebook:
    - 16kHz mono
    - peak-normalize
    - pad/truncate to 4 seconds
    - 80-bin log-mel spectrogram
    - pad/truncate time axis to 200 steps
    - reshape to (1, 80, 200, 1)
    """
    sr = 16000
    target_len_sec = 4.0
    
    # 1. Load and resample
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # 2. Peak normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        
    # 3. Pad/truncate to 4 seconds
    target_len_samples = int(target_len_sec * sr)
    if len(audio) > target_len_samples:
        audio = audio[:target_len_samples]
    elif len(audio) < target_len_samples:
        pad_width = target_len_samples - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')
        
    # 4. Log-mel spectrogram
    n_mels = 80
    n_fft = 1024
    hop_length = int(sr * target_len_sec / 200) # Ensure roughly 200 steps
    # To hit exactly 200 frames, we can use librosa's default hop or custom.
    # We will just pad/truncate the time axis directly later.
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=512, n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 5. Pad/truncate time axis to exactly 200 steps
    target_time_steps = 200
    if log_mel_spec.shape[1] > target_time_steps:
        log_mel_spec = log_mel_spec[:, :target_time_steps]
    elif log_mel_spec.shape[1] < target_time_steps:
        pad_width = target_time_steps - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        
    # 6. Reshape to (1, 80, 200, 1)
    log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)
    log_mel_spec = np.expand_dims(log_mel_spec, axis=0)
    
    return log_mel_spec

def predict(audio_path: str) -> dict:
    if model is None:
        raise RuntimeError("Model not loaded. Please check server logs for initialization errors.")
        
    processed = preprocess_audio(audio_path)
    
    # Prediction
    prediction = model.predict(processed, verbose=0)
    # Assuming output is probability of being a spoof (cloned)
    spoof_prob = float(prediction[0][0])
    
    is_cloned = spoof_prob > 0.5
    verdict = "spoof" if is_cloned else "bonafide"
    confidence = spoof_prob * 100 if is_cloned else (1 - spoof_prob) * 100
    
    return {
        "verdict": verdict,
        "confidence": confidence,
        "is_cloned": is_cloned
    }
