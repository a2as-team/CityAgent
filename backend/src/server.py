import os
import tempfile
import aiofiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.adk.cli.fast_api import get_fast_api_app
from src.rag_pipeline.vector import get_supabase_client, ingest_file, delete_file_chunks

# Directory that contains agent packages (src)
AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
SERVE_WEB_INTERFACE = False  # Set to false to keep backend API-only

ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]


app = FastAPI(title="CityAgent API")

# CORS must be on the *outer app* that serves /rag/*
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

adk_app: FastAPI = get_fast_api_app(
    agents_dir=AGENTS_DIR,
    # session_service_uri= <>,  # sessions are stored in memory for now, and not persisted
    allow_origins=ALLOWED_ORIGINS,
    web=SERVE_WEB_INTERFACE,
)


app.mount("/adk", adk_app)


@app.get("/health")
async def health():
    return {"status": "ok"}


class IngestReq(BaseModel):
    bucket: str
    path: str  # storage_path
    filename: str  # original filename to store in chunk metadata


class DeleteReq(BaseModel):
    file_key: str


ALLOWED_EXT = {".pdf", ".csv", ".xlsx"}


@app.post("/rag/ingest")
async def rag_ingest(req: IngestReq):
    ext = os.path.splitext(req.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Download file bytes from Supabase Storage
    try:
        file_bytes = get_supabase_client().storage.from_(req.bucket).download(req.path)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download from storage: {e}"
        )

    # Write to a temp file asynchronously so your existing vectorizers can read from disk
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)

    try:
        async with aiofiles.open(tmp_path, "wb") as tmp:
            await tmp.write(file_bytes)

        # Use storage path as the file_key so it is unique and stable
        chunks_added = await ingest_file(tmp_path, file_key=req.path)
        return {
            "ok": True,
            "file_key": req.path,
            "chunks_added": chunks_added,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/rag/delete")
async def rag_delete(req: DeleteReq):
    deleted = delete_file_chunks(req.file_key)
    return {"ok": True, "file_key": req.file_key, "chunks_deleted": deleted}
