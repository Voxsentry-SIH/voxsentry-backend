import os
from dotenv import load_dotenv

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
MODEL_PATH = os.getenv("MODEL_PATH", "models/voice_clone_detector.h5")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "models/embedding_extractor.h5")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
