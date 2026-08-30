import os
import logging
import numpy as np
import tensorflow as tf
from app.config import EMBEDDING_MODEL_PATH, CONFIDENCE_THRESHOLD
from app.services.model_service import preprocess_audio

logger = logging.getLogger(__name__)

VOICEPRINTS_DIR = "voiceprints"

# Load model at import time
embedding_model = None
if not os.path.exists(EMBEDDING_MODEL_PATH):
    logger.error(f"Embedding model file not found at {EMBEDDING_MODEL_PATH}. Verification endpoints will fail.")
else:
    try:
        embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH)
        logger.info(f"Loaded embedding extractor model from {EMBEDDING_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load embedding model from {EMBEDDING_MODEL_PATH}: {str(e)}")

def get_embedding(audio_path: str) -> np.ndarray:
    if embedding_model is None:
        raise RuntimeError("Embedding model not loaded. Please check server logs for initialization errors.")
        
    # Reusing the same preprocessing logic
    processed = preprocess_audio(audio_path)
    
    # Extract embedding
    embedding = embedding_model.predict(processed, verbose=0)
    return embedding[0] # assuming shape (1, embedding_dim)

def enroll_voice(user_id: str, profile_name: str, sample_paths: list) -> dict:
    if not os.path.exists(VOICEPRINTS_DIR):
        os.makedirs(VOICEPRINTS_DIR)
        
    embeddings = []
    for path in sample_paths:
        emb = get_embedding(path)
        embeddings.append(emb)
        
    # Average the embeddings for a robust profile
    profile_embedding = np.mean(embeddings, axis=0)
    
    # Normalize the profile embedding
    profile_embedding = profile_embedding / np.linalg.norm(profile_embedding)
    
    file_path = os.path.join(VOICEPRINTS_DIR, f"{user_id}_{profile_name}.npy")
    np.save(file_path, profile_embedding)
    
    return {
        "success": True, 
        "message": f"Successfully enrolled profile '{profile_name}'.",
        "voice_id": f"{user_id}_{profile_name}"
    }

def verify_voice(test_audio_path: str, user_id: str, profile_name: str, threshold: float = None) -> dict:
    if threshold is None:
        threshold = CONFIDENCE_THRESHOLD
        
    file_path = os.path.join(VOICEPRINTS_DIR, f"{user_id}_{profile_name}.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Voiceprint for {user_id}_{profile_name} not found.")
        
    profile_embedding = np.load(file_path)
    
    test_embedding = get_embedding(test_audio_path)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)
    
    # Cosine similarity
    similarity = np.dot(profile_embedding, test_embedding)
    
    match = similarity >= threshold
    
    return {
        "match": bool(match),
        "similarity_score": float(similarity * 100),
        "message": "Voice verified successfully." if match else "Voice does not match the enrolled profile."
    }
