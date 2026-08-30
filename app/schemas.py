from pydantic import BaseModel

class AnalyzeResponse(BaseModel):
    verdict: str
    confidence: float
    is_cloned: bool

class EnrollResponse(BaseModel):
    success: bool
    message: str
    voice_id: str

class VerifyResponse(BaseModel):
    match: bool
    similarity_score: float
    message: str
