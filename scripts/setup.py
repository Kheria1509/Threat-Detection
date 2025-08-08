#!/usr/bin/env python3
"""
Setup script for Threat Detection System
Installs dependencies and prepares the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "data/models",
        "data/samples", 
        "assets/images",
        "assets/videos",
        "logs",
        "outputs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_models():
    """Check if model files exist"""
    model_files = [
        "data/models/fire.onnx",
        "data/models/weapon1.onnx", 
        "data/models/weapon2.onnx",
        "data/models/mask.onnx",
        "data/models/video_classifier_model_lstm.h5",
        "data/models/maskdetection.keras"
    ]
    
    missing_models = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        print("⚠️  Missing model files:")
        for model in missing_models:
            print(f"   - {model}")
        print("Please ensure all model files are in the data/models directory")
    else:
        print("✅ All model files found!")

def main():
    print("🚀 Setting up Threat Detection System...")
    print("=" * 50)
    
    # Install dependencies
    if not install_requirements():
        return
    
    # Create directories
    create_directories()
    
    # Check models
    check_models()
    
    print("=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python scripts/run_detection.py basic")
    print("2. Or run: python scripts/run_detection.py web")
    print("3. Check the README.md for more options")

if __name__ == "__main__":
    main() 