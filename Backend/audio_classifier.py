"""
audio_classifier.py - Respiratory Disease Classification using .keras model
Uses the EXACT preprocessing pipeline from training
"""

import os
import numpy as np
import librosa
import logging
from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RespiratoryAudioClassifier:
    """Audio classifier for respiratory diseases with training-matched preprocessing"""
    
    DISEASE_CLASSES = [
        "Bronchiectasis",
        "Bronchiolitis", 
        "COPD",
        "Healthy",
        "Pneumonia",
        "URTI"
    ]
    
    SEVERITY_THRESHOLDS = {
        "high": 0.90,
        "moderate": 0.75,
        "mild": 0.60
    }
    
    def __init__(
        self, 
        model_path: str = "models/respiratory_classifier.keras",
        sample_rate: int = 22050,
        duration: float = 20.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        target_length: int = 432
    ):
        """
        Initialize audio classifier with EXACT training parameters
        
        Args:
            model_path: Path to trained .keras model
            sample_rate: Audio sample rate (22050 from training)
            duration: Audio duration in seconds (20.0 from training)
            n_mels: Number of mel bands (128 from training)
            n_fft: FFT window size (2048 from training)
            hop_length: Hop length for spectrogram (512 from training)
            target_length: Target time steps (432 from training)
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
        
        self.model = None
        self.load_model()
        logger.info(f"RespiratoryAudioClassifier initialized with training parameters")
        logger.info(f"Sample Rate: {sample_rate}, Duration: {duration}s, Mel Bands: {n_mels}")
    
    def load_model(self):
        """Load the trained Keras model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from: {self.model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_audio(self, file_path: str, offset: float = 0.5) -> np.ndarray:
        """
        Load audio file with EXACT parameters from training
        
        Args:
            file_path: Path to audio file
            offset: Start offset in seconds (0.5 from training)
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            audio, _ = librosa.load(
                file_path, 
                sr=self.sample_rate,
                duration=self.duration, 
                offset=offset,
                res_type='kaiser_fast'
            )
            logger.info(f"Audio loaded: {file_path}, shape: {audio.shape}")
            return audio
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return np.zeros(int(self.duration * self.sample_rate))
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram with EXACT parameters from training
        Maintains 2D structure (n_mels, target_length)
        
        Args:
            audio: Audio waveform
            
        Returns:
            Normalized mel-spectrogram (n_mels, target_length)
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0
            )
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            if mel_spec_norm.shape[1] < self.target_length:
                pad_width = self.target_length - mel_spec_norm.shape[1]
                mel_spec_norm = np.pad(
                    mel_spec_norm, 
                    ((0, 0), (0, pad_width)), 
                    mode='constant', 
                    constant_values=0
                )
            elif mel_spec_norm.shape[1] > self.target_length:
                mel_spec_norm = mel_spec_norm[:, :self.target_length]
            
            logger.info(f"Mel-spectrogram extracted, shape: {mel_spec_norm.shape}")
            return mel_spec_norm
            
        except Exception as e:
            logger.error(f"Error extracting mel-spectrogram: {e}")
            return np.zeros((self.n_mels, self.target_length))
    
    def preprocess_for_inference(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio for model inference (NO augmentation during inference)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed mel-spectrogram ready for model input
        """
        audio = self.load_audio(audio_path)
        mel_spec = self.extract_mel_spectrogram(audio)
        mel_spec_expanded = np.expand_dims(mel_spec, axis=-1)
        mel_spec_batch = np.expand_dims(mel_spec_expanded, axis=0)
        
        logger.info(f"Preprocessed for inference, final shape: {mel_spec_batch.shape}")
        return mel_spec_batch
    
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """
        Predict respiratory disease with minimal response for API
        """
        try:
            logger.info(f"Starting prediction for: {audio_path}")
            
            features = self.preprocess_for_inference(audio_path)
            predictions = self.model.predict(features, verbose=0)
            
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_disease = self.DISEASE_CLASSES[predicted_class_idx]
            severity = self._calculate_severity(confidence, predicted_disease)
            
            result = {
                "disease": predicted_disease,
                "confidence": confidence,
                "severity": severity
            }
            
            logger.info(f"Prediction: {predicted_disease} (Confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def _calculate_severity(self, confidence: float, disease: str) -> str:
        """
        Calculate severity based on confidence score and disease type
        
        Args:
            confidence: Prediction confidence (0.0-1.0)
            disease: Predicted disease name
            
        Returns:
            Severity level string
        """
        if disease == "Healthy":
            return "none"
        
        if confidence >= self.SEVERITY_THRESHOLDS["high"]:
            return "high"
        elif confidence >= self.SEVERITY_THRESHOLDS["moderate"]:
            return "moderate"
        elif confidence >= self.SEVERITY_THRESHOLDS["mild"]:
            return "mild"
        else:
            return "uncertain"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "total_params": self.model.count_params(),
            "disease_classes": self.DISEASE_CLASSES,
            "preprocessing_params": {
                "sample_rate": self.sample_rate,
                "duration": self.duration,
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "target_length": self.target_length
            }
        }


if __name__ == "__main__":
    classifier = RespiratoryAudioClassifier(
        model_path="models/respiratory_classifier.keras",
        sample_rate=22050,
        duration=20.0,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        target_length=432
    )
    
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    model_info = classifier.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")
    print("="*70)
    
    test_audio = "path/to/test_audio.wav"
    
    if os.path.exists(test_audio):
        result = classifier.predict(test_audio)
        
        print("\n" + "="*70)
        print("RESPIRATORY DISEASE CLASSIFICATION")
        print("="*70)
        print(f"Disease: {result['disease']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Severity: {result['severity']}")
        print("="*70)
    else:
        print(f"\nTest audio file not found: {test_audio}")
        print("Please provide a valid audio file path to test the classifier.")
