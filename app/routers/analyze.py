import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import AnalyzeResponse
from app.services.model_service import predict

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        try:
            result = predict(temp_file.name)
            return AnalyzeResponse(**result)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
