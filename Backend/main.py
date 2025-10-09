"""
main.py - FastAPI Backend for A.I.R.A. (AI Respiratory Assistant)
Connects Audio Classifier, LLM Agent, and Database for frontend integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi import FastAPI, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any, List
import os
import logging
from datetime import datetime, timedelta
import uuid
import shutil
from contextlib import asynccontextmanager
from jose import jwt
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

    logger.info("🚀 A.I.R.A Backend starting up...")
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

    logger.info("🧹 A.I.R.A Backend shutting down...")
    audio_classifier = None

    yield

    logger.info("🧹 A.I.R.A AI Backend shutting down...")
    if chatbot:
        try:
            chatbot.cleanup()
        except:
            pass
        try:
            chatbot.cleanup()
        except:
            pass
    logger.info("✓ Cleanup complete")


# ---------------- FASTAPI APP ---------------- #
app = FastAPI(
    title="A.I.R.A API",
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

# ---------------- FASTAPI APP ---------------- #
app = FastAPI(
    title="A.I.R.A API",
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
    avatar: Optional[str] = None

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
    confidence: Optional[float] = None
    disease_classification: Optional[str] = None
    follow_up: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

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
    avatar: Optional[str] = None

# ---------------- ENDPOINTS ---------------- #
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "A.I.R.A API",
        "version": "1.0.1",
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

# ---------------- HEALTH CHECK ---------------- #
@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    global chatbot, audio_classifier
    
    return { 
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "services": {
            "chatbot": "available" if chatbot else "unavailable",
            "audio_classifier": "available" if audio_classifier else "unavailable"
        }
    }

# ---------------- AUTHENTICATION ---------------- #
@app.post("/register", tags=["Authentication"])
async def register_patient(
    patient_id: str = Form(...),
    age_range: str = Form(...),
    gender: str = Form(...),
    smoking_status: str = Form(...),
    has_hypertension: bool = Form(False),
    has_diabetes: bool = Form(False),
    has_asthma_history: bool = Form(False),
    previous_respiratory_infections: int = Form(0),
    current_medications: str = Form(""),
    allergies: str = Form(""),
    last_consultation_date: Optional[str] = Form(None),
    avatar: UploadFile = File(None)
):
    """Register a new patient with optional avatar upload"""
    
    avatar_path = None
    if avatar:
        # Validate file type
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        file_ext = os.path.splitext(avatar.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format. Allowed formats: {', '.join(allowed_extensions)}"
            )
        
        # Save avatar
        os.makedirs("uploads/avatars", exist_ok=True)
        avatar_filename = f"{patient_id}_{uuid.uuid4()}{file_ext}"
        avatar_path = os.path.join("uploads/avatars", avatar_filename)
        
        with open(avatar_path, "wb") as buffer:
            shutil.copyfileobj(avatar.file, buffer)
        
        avatar_path = f"/uploads/avatars/{avatar_filename}"
    
    db_data = {
        'patient_id': patient_id,
        'age_range': age_range,
        'gender': gender,
        'smoking_status': smoking_status,
        'has_hypertension': has_hypertension,
        'has_diabetes': has_diabetes,
        'has_asthma_history': has_asthma_history,
        'previous_respiratory_infections': previous_respiratory_infections,
        'current_medications': current_medications,
        'allergies': allergies,
        'last_consultation_date': last_consultation_date,
        'avatar': avatar_path
    }
    
    agent = DataRetrievalAgent()
    try:
        success = agent.db_manager.create_patient(db_data)
        if not success:
            if avatar_path and os.path.exists(avatar_path.lstrip('/')):
                os.remove(avatar_path.lstrip('/'))
            raise HTTPException(
                status_code=409,
                detail="Patient with this ID already exists. Please choose a different ID."
            )
        
        logger.info(f"New patient registered: {patient_id}")
        return JSONResponse(
            content={
                "message": "Patient registered successfully! You can now login with your Patient ID.",
                "patient_id": patient_id,
                "avatar": avatar_path
            },
            status_code=201
        )
    except HTTPException:
        raise
    except Exception as e:
        if avatar_path and os.path.exists(avatar_path.lstrip('/')):
            os.remove(avatar_path.lstrip('/'))
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )
    finally:
        agent.cleanup()

# ---------------- AUTHENTICATION ---------------- #
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
        
        avatar_url = patient_profile.get("avatar") or "https://via.placeholder.com/150/4A90E2/FFFFFF?text=Patient"
        
        logger.info(f"Patient logged in: {form_data.username}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_info={
                "username": patient_profile.get("patient_id"),
                "name": f"Patient {patient_profile.get('patient_id')[:8]}...",
                "avatar": avatar_url,
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

# ---------------- API ENDPOINTS ---------------- #
@app.post("/api/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_ai(request: ChatRequest):
    """Handles chat interactions with the AI assistant."""
    global chatbot
    
    if not chatbot:
        raise HTTPException(
            status_code=503, 
            detail="Chatbot service not available. Please check backend logs."
        )
    
    try:
        logger.info(f"Chat request from patient: {request.patient_id[:8]}...")
        
        response = chatbot.query(
            user_query=request.query,
            patient_id=request.patient_id,
            audio_result=request.audio_result
        )
        
        return ChatResponse(**response)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Chat failed: {str(e)}"
        )

# ---------------- AUDIO ANALYSIS ---------------- #
@app.post("/api/analyze-audio", tags=["Audio Analysis"])
async def analyze_respiratory_audio(
    patient_id: str = Form(...), 
    user_query: str = Form(""),
    file: UploadFile = File(...)
):
    """Analyzes a respiratory audio file for disease classification."""
    global audio_classifier
    
    if not audio_classifier:
        raise HTTPException(
            status_code=503, 
            detail="Audio classifier not available. Please check if the model file exists."
        )
    
    allowed_extensions = ['.wav', '.mp3', '.flac']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid audio file format. Allowed formats: {', '.join(allowed_extensions)}"
        )

    file_id = str(uuid.uuid4())
    file_path = os.path.join("uploads/audio", f"{file_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Analyzing audio file for patient: {patient_id[:8]}...")
        
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

# ---------------- COMBINED WORKFLOW ---------------- #
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
        
        return {
            "classification_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Audio analysis failed: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

# ---------------- PATIENT DATA MANAGEMENT ---------------- #
@app.get("/api/patient/{patient_id}", response_model=PatientDataResponse, tags=["Patient Data"])
async def get_patient_data(patient_id: str):
    """Retrieves medical information for a specific patient."""
    agent = DataRetrievalAgent()
    try:
        patient_data = agent.db_manager.get_patient_data(patient_id)
        if not patient_data:
            raise HTTPException(
                status_code=404, 
                detail="Patient not found"
            )
        
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
        logger.error(f"Get patient data error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve patient data: {str(e)}"
        )
    finally:
        agent.cleanup()

# ---------------- PATIENT HISTORY ---------------- #
@app.get("/api/patient/{patient_id}/history", tags=["Patient Data"])
async def get_patient_history(patient_id: str, limit: int = 5):
    """Retrieves recent medical history for a patient."""
    agent = DataRetrievalAgent()
    try:
        history = agent.db_manager.get_patient_history(patient_id, limit)
        return {
            "patient_id": patient_id, 
            "history_count": len(history), 
            "history": history
        }
    except Exception as e:
        logger.error(f"Get patient history error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve history: {str(e)}"
        )
    finally:
        agent.cleanup()

# ---------------- MODEL INFO ---------------- #
@app.get("/api/model/info", tags=["Model Information"])
async def get_model_info():
    """Gets information about the loaded audio classification model."""
    global audio_classifier
    
    if not audio_classifier:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    try:
        return JSONResponse(content=audio_classifier.get_model_info())
    except Exception as e:
        logger.error(f"Get model info error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get model info: {str(e)}"
        )

# ---------------- UPDATE PATIENT PROFILE ---------------- #
@app.put("/api/patient/{username}/update", tags=["Patient Data"])
async def update_patient_profile(
    username: str,
    age_range: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    smoking_status: Optional[str] = Form(None),
    has_hypertension: Optional[bool] = Form(None),
    has_diabetes: Optional[bool] = Form(None),
    has_asthma_history: Optional[bool] = Form(None),
    previous_respiratory_infections: Optional[int] = Form(None),
    current_medications: Optional[str] = Form(None),
    allergies: Optional[str] = Form(None),
    last_consultation_date: Optional[str] = Form(None),
    avatar: Optional[UploadFile] = File(None)
):
    """Update patient profile information including avatar"""
    agent = DataRetrievalAgent()
    try:
        patient = agent.db_manager.get_patient_data(username)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        update_data = {}
        
        form_data = {
            "age_range": age_range, "gender": gender, "smoking_status": smoking_status,
            "has_hypertension": has_hypertension, "has_diabetes": has_diabetes,
            "has_asthma_history": has_asthma_history, 
            "previous_respiratory_infections": previous_respiratory_infections,
            "current_medications": current_medications, "allergies": allergies,
            "last_consultation_date": last_consultation_date
        }

        for key, value in form_data.items():
            if value is not None:
                update_data[key] = value

        if avatar:
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            file_ext = os.path.splitext(avatar.filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image format. Allowed: {', '.join(allowed_extensions)}"
                )
            
            os.makedirs("uploads/avatars", exist_ok=True)
            
            old_avatar = patient.get('avatar')
            if old_avatar and os.path.exists(old_avatar.lstrip('/')):
                try:
                    os.remove(old_avatar.lstrip('/'))
                except Exception as e:
                    logger.warning(f"Could not delete old avatar: {e}")
            
            avatar_filename = f"{username}_{uuid.uuid4()}{file_ext}"
            avatar_path = os.path.join("uploads/avatars", avatar_filename)
            
            with open(avatar_path, "wb") as buffer:
                shutil.copyfileobj(avatar.file, buffer)
            
            update_data["avatar"] = f"/uploads/avatars/{avatar_filename}"

        if not update_data:
             return JSONResponse(content={"message": "No fields to update"}, status_code=200)

        success = agent.db_manager.update_patient(username, update_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update patient profile")

        return JSONResponse(content={
            "message": "Profile updated successfully",
            "updated_fields": list(update_data.keys())
        }, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        agent.cleanup()

# ---------------- STATIC FILES ---------------- #
os.makedirs("uploads/avatars", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

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
    
    logger.info("Starting A.I.R.A Backend Server...")
    logger.info("API Documentation available at: http://localhost:8000/docs")
    
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )