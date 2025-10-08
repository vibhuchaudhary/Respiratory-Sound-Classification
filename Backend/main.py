"""
main.py - FastAPI Backend for LungScope AI Healthcare Assistant
Connects Audio Classifier, LLM Agent, and Database for frontend integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import logging
from datetime import datetime, timedelta
import uuid
import shutil
from contextlib import asynccontextmanager
from jose import jwt

from llm_agent import LungScopeChatbot
from database import DataRetrievalAgent
from audio_classifier import RespiratoryAudioClassifier

# ---------------- JWT CONFIG ---------------- #
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ---------------- LOGGING ---------------- #
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lungscope_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- GLOBAL VARIABLES ---------------- #
chatbot = None
audio_classifier = None

# ---------------- LIFESPAN HANDLER ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all agents and resources on startup; clean up on shutdown."""
    global chatbot, audio_classifier

    logger.info("🚀 LungScope AI Backend starting up...")
    os.makedirs("uploads/audio", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.info("✓ Directories created")

    try:
        chatbot = LungScopeChatbot(temperature=0.3, model_name="gpt-4")
        logger.info("✓ LLM Chatbot initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize chatbot: {e}")
        chatbot = None

    try:
        audio_classifier = RespiratoryAudioClassifier(
            model_path="models/trained_model.keras",
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
        audio_classifier = None

    yield

    logger.info("🧹 LungScope AI Backend shutting down...")
    if chatbot:
        try:
            chatbot.cleanup()
        except:
            pass
    logger.info("✓ Cleanup complete")


# ---------------- FASTAPI APP ---------------- #
app = FastAPI(
    title="LungScope AI API",
    description="AI Healthcare Assistant for Respiratory Analysis and Smart Diagnostics",
    version="1.0.2",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ---------------- MIDDLEWARE ---------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PYDANTIC MODELS ---------------- #
class PatientCreate(BaseModel):
    patient_id: str
    age_range: str
    gender: str
    smoking_status: str
    has_hypertension: bool
    has_diabetes: bool
    has_asthma_history: bool
    previous_respiratory_infections: int
    current_medications: str
    allergies: str
    last_consultation_date: Optional[str] = None

class UserLogin(BaseModel):
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_info: Dict[str, Any]

class ChatRequest(BaseModel):
    patient_id: str = Field(..., example="PATIENT_12345")
    query: str = Field(..., example="What should I do about my breathing difficulty?")
    audio_result: Optional[Dict[str, Any]] = Field(None, example={
        "disease": "COPD",
        "confidence": 0.9759,
        "severity": "moderate"
    })

class ChatResponse(BaseModel):
    response: str
    confidence: float
    disease_classification: Optional[str] = None
    follow_up: str
    timestamp: str

class AudioAnalysisResponse(BaseModel):
    disease: str
    confidence: float
    severity: str
    timestamp: str

class PatientDataResponse(BaseModel):
    patient_id: str
    age_range: Optional[str]
    gender: Optional[str]
    smoking_status: Optional[str]
    comorbidities: List[str]
    current_medications: Optional[str]
    allergies: Optional[str]

# ---------------- ENDPOINTS ---------------- #

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "🏥 LungScope AI API",
        "version": "1.0.2",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "register": "/register",
            "login": "/login",
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


@app.post("/register", tags=["Authentication"])
async def register_patient(patient_data: PatientCreate):
    """Register a new patient (no password required)"""
    db_data = patient_data.model_dump()
    
    agent = DataRetrievalAgent()
    try:
        success = agent.db_manager.create_patient(db_data)
        if not success:
            raise HTTPException(
                status_code=409,
                detail="Patient with this ID already exists. Please choose a different ID."
            )
        
        logger.info(f"New patient registered: {patient_data.patient_id}")
        return JSONResponse(
            content={
                "message": "Patient registered successfully! You can now login with your Patient ID.",
                "patient_id": patient_data.patient_id
            },
            status_code=201
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )
    finally:
        agent.cleanup()


@app.post("/login", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: UserLogin):
    """Login with just username/patient_id (no password required)"""
    agent = DataRetrievalAgent()
    try:
        user = agent.db_manager.get_patient_for_auth(form_data.username)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Patient ID not found. Please check your ID or register."
            )
        
        patient_profile = agent.db_manager.get_patient_data(form_data.username)
        if not patient_profile:
            raise HTTPException(
                status_code=500,
                detail="Failed to load patient profile."
            )
        
        access_token = create_access_token(data={"sub": user["patient_id"]})
        
        logger.info(f"Patient logged in: {form_data.username}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_info={
                "username": patient_profile.get("patient_id"),
                "name": f"Patient {patient_profile.get('patient_id')[:8]}...",
                "avatar": "https://via.placeholder.com/150/4A90E2/FFFFFF?text=Patient",
                "age_range": patient_profile.get("age_range"),
                "gender": patient_profile.get("gender")
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Login failed: {str(e)}"
        )
    finally:
        agent.cleanup()


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
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .flac)"),
    patient_id: str = Form(..., description="Patient identifier")
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
    file: UploadFile = File(..., description="Respiratory audio file"),
    patient_id: str = Form(..., description="Patient identifier"),
    query: str = Form(..., description="Patient's medical question")
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


@app.get("/api/patient/{patient_id}", response_model=PatientDataResponse, tags=["Patient Data"])
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


# ---------------- ERROR HANDLERS ---------------- #
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "path": str(request.url),
            "available_endpoints": "/docs"
        }
    )


# ---------------- MAIN EXECUTION BLOCK ---------------- #
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting LungScope AI Backend Server...")
    logger.info("API Documentation available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )