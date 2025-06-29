"""
Configuration utility for Threat Detection System
Manages model paths, settings, and parameters
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the threat detection system"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = DATA_DIR / "models"
    ASSETS_DIR = BASE_DIR / "assets"
    
    # Model paths
    FIRE_SMOKE_MODEL = DATA_DIR / "detect" / "fire_smoke_train" / "weights" / "best.pt"
    WEAPON_MODEL_1 = DATA_DIR / "detect" / "weapondetction1_train" / "weights" / "best.pt"
    WEAPON_MODEL_2 = DATA_DIR / "detect" / "weapon2_detection_train" / "weights" / "best.pt"
    GENERAL_MODEL = DATA_DIR / "detect" / "train" / "weights" / "best.pt"
    
    # ONNX models
    FIRE_ONNX = MODELS_DIR / "fire.onnx"
    WEAPON1_ONNX = MODELS_DIR / "weapon1.onnx"
    WEAPON2_ONNX = MODELS_DIR / "weapon2.onnx"
    MASK_ONNX = MODELS_DIR / "mask.onnx"
    
    # Keras models
    LSTM_MODEL = MODELS_DIR / "video_classifier_model_lstm.h5"
    VIOLENCE_MODEL = DATA_DIR / "detect" / "modelnew.h5"
    MASK_KERAS = MODELS_DIR / "maskdetection.keras"
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.4
    NMS_THRESHOLD = 0.5
    IMAGE_SIZE = (640, 480)
    
    # Video settings
    FPS = 30
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Web interface settings
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 5000
    WEB_DEBUG = True
    
    @classmethod
    def get_model_paths(cls):
        """Get all available model paths"""
        return {
            "fire_smoke": str(cls.FIRE_SMOKE_MODEL),
            "weapon1": str(cls.WEAPON_MODEL_1),
            "weapon2": str(cls.WEAPON_MODEL_2),
            "general": str(cls.GENERAL_MODEL),
            "fire_onnx": str(cls.FIRE_ONNX),
            "weapon1_onnx": str(cls.WEAPON1_ONNX),
            "weapon2_onnx": str(cls.WEAPON2_ONNX),
            "mask_onnx": str(cls.MASK_ONNX),
            "lstm": str(cls.LSTM_MODEL),
            "violence": str(cls.VIOLENCE_MODEL),
            "mask_keras": str(cls.MASK_KERAS)
        }
    
    @classmethod
    def check_models_exist(cls):
        """Check if all model files exist"""
        missing_models = []
        model_paths = cls.get_model_paths()
        
        for name, path in model_paths.items():
            if not os.path.exists(path):
                missing_models.append((name, path))
        
        return missing_models
    
    @classmethod
    def get_class_names(cls):
        """Get class names for different models"""
        return {
            "fire_smoke": ["fire", "smoke"],
            "weapon": ["weapon"],
            "general": ["masked", "person", "masked"],
            "violence": ["normal", "violence"]
        } 