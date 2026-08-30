# VoxSentry Backend

This is the FastAPI backend for the VoxSentry voice clone detection app, built using TensorFlow.

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Files:**
   The backend expects two trained `.h5` model files (usually exported from Kaggle/Colab notebooks). You must place them in the `models/` directory:
   - `models/voice_clone_detector.h5`
   - `models/embedding_extractor.h5`

3. **Environment Variables:**
   Copy the `.env.example` file to `.env` and configure your paths and thresholds.
   ```bash
   cp .env.example .env
   ```

## Running the Server

Run the server locally with `uvicorn`:
```bash
uvicorn app.main:app --reload --port 8000
```
The server will start on `http://localhost:8000`. You can test the health endpoint at `http://localhost:8000/health`.

## Deployment (Render)

This repository includes a `render.yaml` file for easy deployment to [Render](https://render.com) (or similar services like Railway).

1. Push this repository to GitHub.
2. Connect your GitHub repository in the Render dashboard and create a new **Web Service**. It will automatically detect the `render.yaml`.
3. In the Render dashboard, set the required Environment Variables:
   - `MODEL_PATH`: e.g., `models/voice_clone_detector.h5`
   - `EMBEDDING_MODEL_PATH`: e.g., `models/embedding_extractor.h5`
   - `FRONTEND_URL`: The URL of your deployed Next.js frontend (e.g., `https://voxsentry.vercel.app`).

**CRITICAL NOTE ON MODEL FILES (.h5):**
Free tier cloud instances and GitHub have strict file size limits (GitHub caps files at 100MB unless using Git LFS). If your `.h5` files are too large, pushing them to GitHub will fail. 

To deploy successfully:
- **Option A (If files < 100MB):** Simply commit the `.h5` files in the `models/` directory and push to GitHub.
- **Option B (If files > 100MB):** You must either use Git LFS or modify the `render.yaml` build script to download the models from a cloud storage bucket (like AWS S3, Google Cloud Storage, or HuggingFace) before starting the server.
