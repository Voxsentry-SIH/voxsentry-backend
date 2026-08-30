import os
import glob
from typing import List
from fastapi import APIRouter
from app.schemas import VoiceprintResponse
from app.services.embedding_service import VOICEPRINTS_DIR

router = APIRouter()

@router.get("/voiceprints/{user_id}", response_model=List[VoiceprintResponse])
async def get_user_voiceprints(user_id: str):
    if not os.path.exists(VOICEPRINTS_DIR):
        return []
        
    pattern = os.path.join(VOICEPRINTS_DIR, f"{user_id}_*.npy")
    files = glob.glob(pattern)
    
    profiles = []
    # Arbitrary ID generation for frontend mapping
    for idx, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        # Format: {user_id}_{profile_name}.npy
        name_part = filename.replace(f"{user_id}_", "").replace(".npy", "")
        
        # We don't save timestamps/sample counts in the .npy file metadata yet,
        # so we'll return defaults that the frontend can display.
        profiles.append({
            "id": str(idx + 1),
            "name": name_part,
            "samples": 5,
            "date": "Recently"
        })
        
    return profiles
