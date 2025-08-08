# Project Structure Overview

## 📁 Organized Directory Structure

```
Threat-Detection/
├── 📄 README.md                           # Main project documentation
├── 📄 requirements.txt                    # Python dependencies
├── 📄 PROJECT_STRUCTURE.md               # This file - project organization guide
│
├── 📁 src/                               # Source code directory
│   ├── 📁 core/                         # Core detection modules
│   │   ├── 📄 detect.py                 # Basic fire/smoke detection
│   │   ├── 📄 all.py                    # Multi-model detection
│   │   ├── 📄 vio.py                    # Violence detection
│   │   ├── 📄 videodetect.py            # Video detection utilities
│   │   └── 📄 detecting.py              # General detection utilities
│   │
│   ├── 📁 models/                       # AI model implementations
│   │   ├── 📄 lstm.py                   # LSTM-based video classification
│   │   └── 📄 newlstm.py                # Enhanced LSTM model
│   │
│   ├── 📁 web/                          # Web application modules
│   │   ├── 📄 live_feed.py              # Basic web interface
│   │   ├── 📄 improved_live_feed.py     # Enhanced web interface
│   │   └── 📄 vio_live_feed.py          # Violence detection web interface
│   │
│   └── 📁 utils/                        # Utility modules
│       ├── 📄 config.py                 # Configuration management
│       └── 📄 logger.py                 # Logging utilities
│
├── 📁 data/                             # Data and model storage
│   ├── 📁 models/                       # Trained model files
│   │   ├── 📄 fire.onnx                 # Fire detection ONNX model
│   │   ├── 📄 weapon1.onnx              # Weapon detection ONNX model 1
│   │   ├── 📄 weapon2.onnx              # Weapon detection ONNX model 2
│   │   ├── 📄 mask.onnx                 # Mask detection ONNX model
│   │   ├── 📄 video_classifier_model_lstm.h5  # LSTM video classifier
│   │   └── 📄 maskdetection.keras       # Mask detection Keras model
│   │
│   ├── 📁 samples/                      # Sample data for testing
│   │
│   ├── 📁 detect/                       # YOLO training outputs
│   │   ├── 📁 train/                    # General detection training
│   │   ├── 📁 fire_smoke_train/         # Fire/smoke detection training
│   │   ├── 📁 weapon2_detection_train/  # Weapon detection training 2
│   │   ├── 📁 weapondetction1_train/    # Weapon detection training 1
│   │   ├── 📁 lstm/                     # LSTM model files
│   │   ├── 📁 val/                      # Validation results
│   │   └── 📄 modelnew.h5               # Violence detection model
│   │
│   └── 📁 yolo-Weights/                 # YOLO model weights
│       └── 📄 yolov8n.pt                # YOLO v8 nano weights
│
├── 📁 assets/                           # Media assets
│   ├── 📁 images/                       # Image files
│   │   ├── 📄 download.jpeg             # Sample images
│   │   ├── 📄 download1.jpeg
│   │   ├── 📄 download2.jpeg
│   │   ├── 📄 download.jpg
│   │   ├── 📄 images.jpeg
│   │   └── 📄 images1.jpeg
│   │
│   └── 📁 videos/                       # Video files
│       ├── 📄 Battle.mp4                # Sample videos for testing
│       ├── 📄 Brandon.mp4
│       ├── 📄 Explosion.mp4
│       ├── 📄 FIGHT_PRACTICE.mp4
│       ├── 📄 Gun.mp4
│       ├── 📄 Rf.mp4
│       ├── 📄 Road_Fight.mp4
│       ├── 📄 Tian.mp4
│       ├── 📄 V_19.mp4
│       └── 📄 Violence_house.mp4
│
├── 📁 scripts/                          # Utility scripts
│   ├── 📄 run_detection.py              # Main detection runner script
│   └── 📄 setup.py                      # Environment setup script
│
├── 📁 examples/                         # Example implementations
│   ├── 📄 basic_detection_example.py    # Basic detection example
│   └── 📄 web_interface_example.py      # Web interface example
│
├── 📁 notebooks/                        # Jupyter notebooks
│   ├── 📄 image_class.ipynb             # Image classification notebook
│   └── 📄 lstm.ipynb                    # LSTM model notebook
│
├── 📁 docs/                             # Documentation
│   ├── 📄 API_REFERENCE.md              # API documentation
│   └── 📄 DEPLOYMENT.md                 # Deployment guide
│
├── 📁 tests/                            # Test files (to be added)
├── 📁 logs/                             # Log files (created at runtime)
├── 📁 outputs/                          # Output files (created at runtime)
└── 📁 .venv/                            # Virtual environment
```

## 🎯 Key Organizational Benefits

### 1. **Clear Separation of Concerns**
- **`src/core/`**: Main detection logic
- **`src/models/`**: AI model implementations
- **`src/web/`**: Web interface components
- **`src/utils/`**: Shared utilities

### 2. **Logical Data Organization**
- **`data/models/`**: All trained model files
- **`data/detect/`**: Training outputs and weights
- **`assets/`**: Media files for testing

### 3. **Easy Access and Maintenance**
- **`scripts/`**: Quick access to main functionality
- **`examples/`**: Ready-to-run examples
- **`docs/`**: Comprehensive documentation

### 4. **Scalable Structure**
- Modular design for easy expansion
- Clear import paths
- Consistent naming conventions

## 🚀 Quick Access Guide

### **Main Entry Points:**
```bash
# Run different detection modes
python scripts/run_detection.py basic      # Basic detection
python scripts/run_detection.py violence   # Violence detection
python scripts/run_detection.py lstm       # LSTM classification
python scripts/run_detection.py all        # Multi-model detection
python scripts/run_detection.py web        # Web interface

# Setup environment
python scripts/setup.py

# Run examples
python examples/basic_detection_example.py
python examples/web_interface_example.py
```

### **Key Files by Function:**

#### **Detection Systems:**
- `src/core/detect.py` - Basic fire/smoke detection
- `src/core/all.py` - Multi-model detection
- `src/core/vio.py` - Violence detection

#### **AI Models:**
- `src/models/lstm.py` - LSTM video classification
- `data/models/` - All trained model files

#### **Web Interfaces:**
- `src/web/live_feed.py` - Basic web interface
- `src/web/improved_live_feed.py` - Enhanced interface

#### **Configuration:**
- `src/utils/config.py` - Centralized configuration
- `src/utils/logger.py` - Logging system

#### **Documentation:**
- `README.md` - Main project overview
- `docs/API_REFERENCE.md` - API documentation
- `docs/DEPLOYMENT.md` - Deployment guide

## 📋 File Naming Conventions

### **Python Files:**
- **snake_case** for all Python files
- **Descriptive names** indicating functionality
- **Consistent prefixes** for related modules

### **Model Files:**
- **Descriptive names** with model type
- **Version numbers** when applicable
- **Format extensions** (.pt, .h5, .onnx, .keras)

### **Asset Files:**
- **Descriptive names** for media files
- **Organized by type** (images/, videos/)
- **Consistent naming** for related files

## 🔧 Development Workflow

### **Adding New Features:**
1. **Core functionality** → `src/core/`
2. **AI models** → `src/models/`
3. **Web interfaces** → `src/web/`
4. **Utilities** → `src/utils/`

### **Adding New Models:**
1. **Model files** → `data/models/`
2. **Training outputs** → `data/detect/`
3. **Configuration** → `src/utils/config.py`

### **Adding Documentation:**
1. **API docs** → `docs/API_REFERENCE.md`
2. **Deployment guides** → `docs/DEPLOYMENT.md`
3. **Examples** → `examples/`

## 📊 Benefits of This Organization

### **For Developers:**
- ✅ Easy to find relevant code
- ✅ Clear import paths
- ✅ Modular architecture
- ✅ Consistent structure

### **For Users:**
- ✅ Simple setup process
- ✅ Clear usage examples
- ✅ Comprehensive documentation
- ✅ Multiple deployment options

### **For Maintenance:**
- ✅ Logical file grouping
- ✅ Easy to update dependencies
- ✅ Clear separation of concerns
- ✅ Scalable structure

## 🎯 Next Steps

### **Immediate Actions:**
1. **Update import paths** in existing files
2. **Test all functionality** with new structure
3. **Update documentation** with new paths
4. **Create additional examples** as needed

### **Future Enhancements:**
1. **Add unit tests** in `tests/` directory
2. **Create CI/CD pipeline** configuration
3. **Add Docker support** for containerization
4. **Implement monitoring** and logging

This organized structure makes the project more professional, maintainable, and user-friendly while preserving all existing functionality! 