import base64                                                                                                         
import os                                                                                                             
import tempfile                                                                                                       
import threading                                                                                                      
import cv2  
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI, 
    File, 
    HTTPException, 
    UploadFile,
)  
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from golfswing.model import load_trained_model                                                                                                                                                                 
from golfswing.predict import (
    TestVideo, 
    compute_event_frames, 
    extract_frames
)


MODEL_PATH = "models/20261108-025410_best.weights.h5"
state = {}

@asynccontextmanager
async def lifespan(app):
    state['model'] = load_trained_model(MODEL_PATH)
    yield 


app = FastAPI(lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend" / "dist"

app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")

EVENT_NAMES = [
    "Address", "Toe-up", "Mid-backswing", "Top",
    "Mid-downswing", "Impact", "Mid-follow-through", "Finish"
]

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024 
model_lock = threading.Lock()

@app.post("/api/analyze")
def analyze(video: UploadFile = File(...)):
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext or 'unknown'}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        size = 0
        with tmp:
            while chunk := video.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video is too large (100 MB max)")
                tmp.write(chunk)
        frames = TestVideo(tmp.name).frames
        with model_lock:
            event_frames = compute_event_frames(state['model'], frames)
        images = extract_frames(tmp.name, event_frames)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    finally:
        os.remove(tmp.name)
    results = []
    for name, idx, img in zip(EVENT_NAMES, event_frames, images):
        _, buf = cv2.imencode(".jpg", img)
        results.append({
            "event": name,
            "frame": int(idx),
            "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        })
    return {"events": results}