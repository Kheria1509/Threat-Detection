# API Reference

## Core Detection Modules

### Basic Detection (`src/core/detect.py`)

Main fire/smoke detection module using YOLO.

**Functions:**
- `main()`: Runs the basic detection system
- `detect_fire_smoke(frame)`: Detects fire and smoke in a frame

**Usage:**
```python
from src.core.detect import main
main()
```

### Multi-Model Detection (`src/core/all.py`)

Combines multiple detection models for comprehensive threat analysis.

**Functions:**
- `main()`: Runs all detection models simultaneously
- `process_frame(frame)`: Processes a single frame with all models

**Usage:**
```python
from src.core.all import main
main()
```

### Violence Detection (`src/core/vio.py`)

LSTM-based violence detection in video streams.

**Functions:**
- `print_results(model_path)`: Runs violence detection on video
- `detect_violence(frame)`: Detects violence in a single frame

**Usage:**
```python
from src.core.vio import print_results
print_results("./data/models/modelnew.h5")
```

## Model Modules

### LSTM Models (`src/models/`)

LSTM-based video classification models.

**Files:**
- `lstm.py`: Main LSTM implementation
- `newlstm.py`: Enhanced LSTM model

**Usage:**
```python
from src.models.lstm import main
main()
```

## Web Interface (`src/web/`)

Flask-based web applications for remote monitoring.

**Files:**
- `live_feed.py`: Basic live feed interface
- `improved_live_feed.py`: Enhanced interface with multiple models
- `vio_live_feed.py`: Violence detection web interface

**Usage:**
```python
from src.web.live_feed import app
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Utility Modules (`src/utils/`)

### Configuration (`src/utils/config.py`)

Centralized configuration management.

**Class:**
- `Config`: Configuration class with model paths and settings

**Usage:**
```python
from src.utils.config import Config

# Get model paths
paths = Config.get_model_paths()

# Check if models exist
missing = Config.check_models_exist()

# Get class names
classes = Config.get_class_names()
```

### Logging (`src/utils/logger.py`)

Centralized logging functionality.

**Class:**
- `ThreatDetectionLogger`: Logger class for the system

**Usage:**
```python
from src.utils.logger import logger

logger.info("System started")
logger.detection_event("weapon", 0.95, "camera_1")
logger.error("Detection failed")
```

## Scripts (`scripts/`)

### Main Runner (`scripts/run_detection.py`)

Main script to run different detection modes.

**Usage:**
```bash
python scripts/run_detection.py basic
python scripts/run_detection.py violence
python scripts/run_detection.py lstm
python scripts/run_detection.py all
python scripts/run_detection.py web
```

### Setup (`scripts/setup.py`)

Setup script for environment preparation.

**Usage:**
```bash
python scripts/setup.py
```

## Examples (`examples/`)

### Basic Detection Example (`examples/basic_detection_example.py`)

Example of basic fire/smoke detection.

**Usage:**
```bash
python examples/basic_detection_example.py
```

### Web Interface Example (`examples/web_interface_example.py`)

Example of running the web interface.

**Usage:**
```bash
python examples/web_interface_example.py
```

## Configuration

### Model Paths

All model paths are managed through the `Config` class:

```python
from src.utils.config import Config

# YOLO Models
fire_smoke_model = Config.FIRE_SMOKE_MODEL
weapon_model_1 = Config.WEAPON_MODEL_1
weapon_model_2 = Config.WEAPON_MODEL_2
general_model = Config.GENERAL_MODEL

# ONNX Models
fire_onnx = Config.FIRE_ONNX
weapon1_onnx = Config.WEAPON1_ONNX
weapon2_onnx = Config.WEAPON2_ONNX
mask_onnx = Config.MASK_ONNX

# Keras Models
lstm_model = Config.LSTM_MODEL
violence_model = Config.VIOLENCE_MODEL
mask_keras = Config.MASK_KERAS
```

### Detection Settings

```python
# Confidence threshold for detections
confidence_threshold = Config.CONFIDENCE_THRESHOLD  # 0.4

# Image size for processing
image_size = Config.IMAGE_SIZE  # (640, 480)

# Video settings
fps = Config.FPS  # 30
frame_width = Config.FRAME_WIDTH  # 640
frame_height = Config.FRAME_HEIGHT  # 480
```

### Web Interface Settings

```python
# Web server settings
host = Config.WEB_HOST  # "0.0.0.0"
port = Config.WEB_PORT  # 5000
debug = Config.WEB_DEBUG  # True
```

## Error Handling

### Common Errors

1. **Model Not Found**: Check if model files exist in `data/models/`
2. **Camera Access**: Ensure camera permissions are granted
3. **Memory Issues**: Reduce frame resolution or use GPU acceleration
4. **Import Errors**: Ensure all dependencies are installed

### Logging

All errors and events are logged using the centralized logger:

```python
from src.utils.logger import logger

try:
    # Your code here
    pass
except Exception as e:
    logger.error(f"Error occurred: {e}")
```

## Performance Optimization

### Tips for Better Performance

1. **Use GPU**: Enable CUDA for faster processing
2. **Reduce Resolution**: Lower frame resolution for better FPS
3. **Model Selection**: Use only necessary models
4. **Batch Processing**: Process multiple frames together
5. **Memory Management**: Clear unused variables and models

### Monitoring

Use the logger to monitor system performance:

```python
logger.info(f"Processing FPS: {fps}")
logger.info(f"Detection accuracy: {accuracy:.2f}%")
logger.warning(f"High memory usage: {memory_usage}MB")
``` 