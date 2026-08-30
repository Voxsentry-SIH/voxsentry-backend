import tempfile
import os
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas import EnrollResponse
from app.services.embedding_service import enroll_voice

router = APIRouter()

@router.post("/enroll", response_model=EnrollResponse)
async def enroll_user_voice(
    user_id: str = Form(...), 
    profile_name: str = Form(...), 
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one audio file is required.")
        
    temp_paths = []
    try:
        for file in files:
            if not file.filename.endswith(".wav"):
                raise HTTPException(status_code=400, detail="Only .wav files are supported.")
                
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            temp_paths.append(temp_file.name)
            
        try:
            result = enroll_voice(user_id, profile_name, temp_paths)
            return EnrollResponse(**result)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
