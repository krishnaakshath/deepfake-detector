"""
Deepfake Detection API
FastAPI backend for multi-modal deepfake detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import tempfile
import uuid
from typing import Optional
from datetime import datetime

# Import detection models
from models.video_detector import VideoDeepfakeDetector
from models.audio_detector import AudioDeepfakeDetector
from models.face_analyzer import FaceLandmarkAnalyzer
from models.chatbot import get_chat_response, get_quick_tips
from models.chatbot import get_chat_response, get_quick_tips
from utils.helpers import generate_report, get_protection_tips, validate_file_extension, sanitize_filename, convert_to_serializable
from utils.preprocessing import extract_audio_from_video

# Initialize FastAPI app
app = FastAPI(
    title="DeepGuard - Deepfake Detection API",
    description="AI-powered deepfake detection for videos and audio",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize detectors (lazy loading for performance)
video_detector = None
audio_detector = None
face_analyzer = None


# Pydantic models for request validation
class ChatMessage(BaseModel):
    message: str


def get_video_detector():
    global video_detector
    if video_detector is None:
        video_detector = VideoDeepfakeDetector()
    return video_detector


def get_audio_detector():
    global audio_detector
    if audio_detector is None:
        audio_detector = AudioDeepfakeDetector()
    return audio_detector


def get_face_analyzer():
    global face_analyzer
    if face_analyzer is None:
        face_analyzer = FaceLandmarkAnalyzer()
    return face_analyzer


def cleanup_file(filepath: str):
    """Background task to cleanup uploaded files"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "DeepGuard API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "analyze_video": "/api/analyze/video",
            "analyze_audio": "/api/analyze/audio",
            "analyze_full": "/api/analyze/full",
            "chat": "/api/chat",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "video_detector": "ready",
            "audio_detector": "ready",
            "face_analyzer": "ready"
        }
    }


@app.post("/api/analyze/video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Analyze video for deepfake detection.
    
    Performs:
    - CNN-based visual artifact detection
    - Compression analysis
    - Color consistency checking
    """
    # Validate file
    is_valid, file_type, error = validate_file_extension(file.filename, allowed_video=True, allowed_audio=False)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    safe_filename = sanitize_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze video
        detector = get_video_detector()
        result = detector.analyze_video(filepath)
        
        # Add protection tips
        if result.get('success'):
            verdict = 'likely_fake' if result.get('is_deepfake') else 'likely_authentic'
            result['protection_tips'] = get_protection_tips(verdict)
            result['filename'] = file.filename
        
        return JSONResponse(content=convert_to_serializable(result))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, filepath)


@app.post("/api/analyze/audio")
async def analyze_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Analyze audio for deepfake detection.
    
    Performs:
    - MFCC spectral analysis
    - Voice quality assessment
    - Temporal pattern analysis
    """
    # Validate file
    is_valid, file_type, error = validate_file_extension(file.filename, allowed_video=False, allowed_audio=True)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    safe_filename = sanitize_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze audio
        detector = get_audio_detector()
        result = detector.analyze_audio(filepath)
        
        # Add protection tips
        if result.get('success'):
            verdict = 'likely_fake' if result.get('is_deepfake') else 'likely_authentic'
            result['protection_tips'] = get_protection_tips(verdict)
            result['filename'] = file.filename
        
        return JSONResponse(content=convert_to_serializable(result))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, filepath)


@app.post("/api/analyze/full")
async def analyze_full(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Perform comprehensive multi-modal deepfake analysis.
    
    For videos: Runs video, audio (if present), and face landmark analysis.
    For audio: Runs audio analysis only.
    
    Returns combined report with overall authenticity score.
    """
    # Validate file
    is_valid, file_type, error = validate_file_extension(file.filename, allowed_video=True, allowed_audio=True)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    safe_filename = sanitize_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
    audio_filepath = None
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        video_result = None
        audio_result = None
        face_result = None
        
        if file_type == 'video':
            # Video analysis
            video_detector_instance = get_video_detector()
            video_result = video_detector_instance.analyze_video(filepath)
            
            # Face landmark analysis
            face_analyzer_instance = get_face_analyzer()
            face_result = face_analyzer_instance.analyze_video(filepath)
            
            # Try to extract and analyze audio
            # Skip for likely image files
            is_image = filepath.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'))
            
            if not is_image:
                try:
                    audio_filepath = extract_audio_from_video(filepath)
                    if audio_filepath and os.path.exists(audio_filepath):
                        audio_detector_instance = get_audio_detector()
                        audio_result = audio_detector_instance.analyze_audio(audio_filepath)
                except Exception:
                    # Audio extraction failed, continue without audio analysis
                    audio_result = {'success': False, 'error': 'Could not extract audio from video'}
            else:
                audio_result = {'success': False, 'error': 'Image file - no audio analysis'}
        
        elif file_type == 'audio':
            # Audio-only analysis
            audio_detector_instance = get_audio_detector()
            audio_result = audio_detector_instance.analyze_audio(filepath)
        
        # Generate comprehensive report
        report = generate_report(
            video_result=video_result,
            audio_result=audio_result,
            face_result=face_result,
            filename=file.filename
        )
        
        # Add protection tips based on verdict
        verdict = report['overall_result'].get('verdict', 'uncertain')
        report['protection_tips'] = get_protection_tips(verdict)
        
        return JSONResponse(content=convert_to_serializable(report))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, filepath)
        if audio_filepath:
            background_tasks.add_task(cleanup_file, audio_filepath)


@app.get("/api/tips")
async def get_tips():
    """Get general deepfake protection tips"""
    return {
        "general_tips": [
            "🔍 Look for visual inconsistencies: blurring around face edges, unnatural skin texture, or lighting mismatches",
            "👁️ Check the eyes: deepfakes often have issues with eye movement, blinking patterns, or reflections",
            "👂 Listen carefully: audio deepfakes may have unnatural pauses, robotic tones, or mismatched lip sync",
            "🔎 Verify the source: always check where the media came from and if it's from a trusted source",
            "📱 Use detection tools: tools like this one can help identify potential deepfakes",
            "🧠 Be skeptical: if something seems too good (or bad) to be true, investigate further",
            "📰 Cross-reference: check multiple reliable sources before believing shocking content",
            "🛡️ Protect yourself: be careful about sharing personal photos/videos publicly"
        ],
        "warning_signs": [
            "Unnatural facial movements or expressions",
            "Inconsistent lighting or shadows",
            "Blurry or morphing face edges",
            "Audio that doesn't match lip movements",
            "Unusual blinking patterns (too much or too little)",
            "Strange skin texture or coloring",
            "Background inconsistencies",
            "Jerky or unnatural body movements"
        ],
        "if_you_find_deepfake": [
            "Don't share it further",
            "Report it to the platform",
            "If it's targeting someone, alert them",
            "Document the source if possible",
            "Consider contacting fact-checkers for important content"
        ]
    }


# ==================== AI CHATBOT ENDPOINTS ====================

@app.post("/api/chat")
async def chat_with_bot(chat_message: ChatMessage):
    """
    AI chatbot for deepfake education and guidance.
    
    Send a message and receive educational responses about:
    - What deepfakes are
    - How to detect them
    - How to protect yourself
    - Laws and regulations
    - And more!
    """
    try:
        response = get_chat_response(chat_message.message)
        return JSONResponse(content={
            "success": True,
            "response": response['response'],
            "topic": response['topic'],
            "suggestions": response['suggestions'],
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "response": "I'm sorry, I encountered an error. Please try again!"
        })


@app.get("/api/chat/tips")
async def get_chat_quick_tips():
    """Get quick tips from the chatbot"""
    try:
        tips = get_quick_tips()
        return JSONResponse(content={
            "success": True,
            "tips": tips
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })


@app.get("/api/chat/welcome")
async def get_chat_welcome():
    """Get welcome message for chatbot"""
    return JSONResponse(content={
        "success": True,
        "message": "👋 Hi! I'm **GuardBot**, your AI assistant for deepfake education. I can help you understand deepfakes, learn detection techniques, and stay protected online. What would you like to know?",
        "suggestions": [
            "What is a deepfake?",
            "How can I detect fake videos?",
            "How do I protect myself?"
        ]
    })


# For development: run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

