import tempfile
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas import VerifyResponse
from app.services.embedding_service import verify_voice

router = APIRouter()

@router.post("/verify", response_model=VerifyResponse)
async def verify_user_voice(
    user_id: str = Form(...), 
    profile_name: str = Form(...), 
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        try:
            result = verify_voice(temp_file.name, user_id, profile_name)
            return VerifyResponse(**result)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
