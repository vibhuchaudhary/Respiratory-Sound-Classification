"""
main.py - FastAPI Backend for LungScope AI Healthcare Assistant
Connects Audio Classifier, LLM Agent, and Database for frontend integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import logging
from datetime import datetime
import uuid
import shutil

from llm_agent import LungScopeChatbot
from database import DataRetrievalAgent
from audio_classifier import RespiratoryAudioClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lungscope_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LungScope AI API",
    description="AI Healthcare Assistant for Respiratory Analysis and Smart Diagnostics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None
audio_classifier = None


@app.on_event("startup")
async def startup_event():
    """Initialize all agents and create necessary directories"""
    global chatbot, audio_classifier
    
    logger.info("LungScope AI Backend starting up...")
    
    os.makedirs("uploads/audio", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.info("✓ Directories created")
    
    try:
        chatbot = LungScopeChatbot(temperature=0.3, model_name="gpt-4")
        logger.info("✓ LLM Chatbot initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize chatbot: {e}")
    
    try:
        audio_classifier = RespiratoryAudioClassifier(
            model_path="models/respiratory_classifier.keras",
            sample_rate=22050,
            duration=20.0,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            target_length=432
        )
        logger.info("✓ Audio Classifier initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize audio classifier: {e}")
    
    logger.info("🚀 LungScope AI Backend is ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("LungScope AI Backend shutting down...")
    if chatbot:
        chatbot.cleanup()
    logger.info("✓ Cleanup complete")


class ChatRequest(BaseModel):
    """Chat request from frontend"""
    patient_id: str = Field(..., example="PATIENT_12345")
    query: str = Field(..., example="What should I do about my breathing difficulty?")
    audio_result: Optional[Dict[str, Any]] = Field(None, example={
        "disease": "COPD",
        "confidence": 0.9759,
        "severity": "moderate"
    })


class ChatResponse(BaseModel):
    """Chat response to frontend"""
    response: str
    confidence: float
    disease_classification: Optional[str]
    follow_up: str
    timestamp: str


class AudioAnalysisResponse(BaseModel):
    """Audio analysis response"""
    disease: str
    confidence: float
    severity: str
    timestamp: str


class PatientDataResponse(BaseModel):
    """Patient data response"""
    patient_id: str
    age_range: Optional[str]
    gender: Optional[str]
    smoking_status: Optional[str]
    comorbidities: List[str]
    current_medications: Optional[str]
    allergies: Optional[str]


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "🏥 LungScope AI API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "chat": "/api/chat",
            "audio_analysis": "/api/analyze-audio",
            "full_analysis": "/api/full-analysis",
            "patient_data": "/api/patient/{patient_id}"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "chatbot": chatbot is not None,
            "audio_classifier": audio_classifier is not None and audio_classifier.model is not None,
            "database": True
        }
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_ai(request: ChatRequest):
    """
    Chat with LungScope AI Assistant
    
    Provides personalized medical insights based on:
    - Patient medical history
    - Audio classification results (if available)
    - Medical knowledge base
    """
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service not available")
        
        logger.info(f"Chat request from patient: {request.patient_id[:8]}***")
        
        response = chatbot.query(
            user_query=request.query,
            patient_id=request.patient_id,
            audio_result=request.audio_result
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-audio", response_model=AudioAnalysisResponse, tags=["Audio Analysis"])
async def analyze_respiratory_audio(
    patient_id: str = Field(..., description="Patient identifier"),
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .flac)")
):
    """
    Analyze respiratory audio using trained deep learning model
    
    Returns disease classification with confidence score:
    - Bronchiectasis
    - Bronchiolitis
    - COPD
    - Healthy
    - Pneumonia
    - URTI
    """
    try:
        if not audio_classifier:
            raise HTTPException(status_code=503, detail="Audio classifier not available")
        
        logger.info(f"Audio analysis request from patient: {patient_id[:8]}***")
        
        if not file.filename.endswith(('.wav', '.mp3', '.flac')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported: .wav, .mp3, .flac"
            )
        
        file_id = str(uuid.uuid4())
        file_path = os.path.join("uploads/audio", f"{file_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Audio file saved: {file_path}")
        
        result = audio_classifier.predict(file_path)
        
        logger.info(f"Prediction: {result['disease']} ({result['confidence']:.4f})")
        
        return AudioAnalysisResponse(
            disease=result["disease"],
            confidence=result["confidence"],
            severity=result["severity"],
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in audio analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/full-analysis", tags=["Complete Analysis"])
async def complete_respiratory_analysis(
    patient_id: str = Field(..., description="Patient identifier"),
    query: str = Field(..., description="Patient's medical question"),
    file: UploadFile = File(..., description="Respiratory audio file")
):
    """
    Complete end-to-end analysis workflow:
    1. Classify respiratory disease from audio
    2. Retrieve patient medical history
    3. Get medical knowledge from vector DB
    4. Generate personalized AI response
    
    This is the main endpoint for frontend integration
    """
    try:
        if not audio_classifier or not chatbot:
            raise HTTPException(status_code=503, detail="Services not fully initialized")
        
        logger.info(f"Full analysis request from patient: {patient_id[:8]}***")
        
        if not file.filename.endswith(('.wav', '.mp3', '.flac')):
            raise HTTPException(status_code=400, detail="Invalid audio format")
        
        file_id = str(uuid.uuid4())
        file_path = os.path.join("uploads/audio", f"{file_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        audio_result = audio_classifier.predict(file_path)
        logger.info(f"Audio classified: {audio_result['disease']}")
        
        chat_response = chatbot.query(
            user_query=query,
            patient_id=patient_id,
            audio_result=audio_result
        )
        logger.info(f"Chat response generated")
        
        return JSONResponse(content={
            "status": "success",
            "audio_analysis": {
                "disease": audio_result["disease"],
                "confidence": audio_result["confidence"],
                "severity": audio_result["severity"]
            },
            "ai_response": {
                "response": chat_response["response"],
                "confidence": chat_response["confidence"],
                "follow_up": chat_response["follow_up"]
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in full analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patient/{patient_id}", tags=["Patient Data"])
async def get_patient_data(patient_id: str):
    """
    Retrieve patient medical information from database
    
    Returns anonymized patient data:
    - Demographics (age range, gender)
    - Medical history
    - Comorbidities
    - Medications
    """
    try:
        agent = DataRetrievalAgent()
        patient_data = agent.db_manager.get_patient_data(patient_id)
        agent.cleanup()
        
        if not patient_data:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        comorbidities = []
        if patient_data.get('has_hypertension'):
            comorbidities.append('Hypertension')
        if patient_data.get('has_diabetes'):
            comorbidities.append('Diabetes')
        if patient_data.get('has_asthma_history'):
            comorbidities.append('Asthma History')
        
        return PatientDataResponse(
            patient_id=patient_data['patient_id'],
            age_range=patient_data.get('age_range'),
            gender=patient_data.get('gender'),
            smoking_status=patient_data.get('smoking_status'),
            comorbidities=comorbidities,
            current_medications=patient_data.get('current_medications'),
            allergies=patient_data.get('allergies')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving patient data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patient/{patient_id}/history", tags=["Patient Data"])
async def get_patient_history(patient_id: str, limit: int = 5):
    """
    Retrieve patient medical history
    
    Returns recent medical visits and diagnoses
    """
    try:
        agent = DataRetrievalAgent()
        history = agent.db_manager.get_patient_history(patient_id, limit)
        agent.cleanup()
        
        return JSONResponse(content={
            "patient_id": patient_id,
            "history_count": len(history),
            "history": history
        })
        
    except Exception as e:
        logger.error(f"Error retrieving patient history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info", tags=["Model Information"])
async def get_model_info():
    """
    Get information about the loaded audio classification model
    """
    try:
        if not audio_classifier:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        model_info = audio_classifier.get_model_info()
        return JSONResponse(content=model_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/uploads/cleanup", tags=["Maintenance"])
async def cleanup_old_uploads(days_old: int = 7):
    """
    Clean up old uploaded audio files
    
    Removes audio files older than specified days
    """
    try:
        upload_dir = "uploads/audio"
        deleted_count = 0
        current_time = datetime.now().timestamp()
        
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            file_age_days = (current_time - os.path.getmtime(file_path)) / 86400
            
            if file_age_days > days_old:
                os.remove(file_path)
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old files")
        
        return JSONResponse(content={
            "status": "success",
            "deleted_files": deleted_count,
            "days_threshold": days_old
        })
        
    except Exception as e:
        logger.error(f"Error cleaning up uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
