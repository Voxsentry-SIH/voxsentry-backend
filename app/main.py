from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import FRONTEND_URL
from app.routers import analyze, enroll, verify, voiceprints

app = FastAPI(title="VoxSentry Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(enroll.router, prefix="/api", tags=["Enrollment"])
app.include_router(verify.router, prefix="/api", tags=["Verification"])
app.include_router(voiceprints.router, prefix="/api", tags=["Voiceprints"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
