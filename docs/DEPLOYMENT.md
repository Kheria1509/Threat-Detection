# Deployment Guide

## Quick Start

### 1. Local Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd Threat-Detection

# Install dependencies
pip install -r requirements.txt

# Run setup script
python scripts/setup.py

# Start basic detection
python scripts/run_detection.py basic
```

### 2. Web Interface Setup

```bash
# Start web interface
python scripts/run_detection.py web

# Access at: http://localhost:5000/live_feed
```

## Production Deployment

### Docker Deployment

#### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs outputs

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "scripts/run_detection.py", "web"]
```

#### 2. Build and Run Docker Container

```bash
# Build image
docker build -t threat-detection .

# Run container
docker run -p 5000:5000 --device=/dev/video0:/dev/video0 threat-detection
```

### Cloud Deployment

#### AWS EC2 Deployment

1. **Launch EC2 Instance**
   ```bash
   # Connect to instance
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Update system
   sudo apt-get update
   sudo apt-get upgrade -y
   ```

2. **Install Dependencies**
   ```bash
   # Install Python and pip
   sudo apt-get install python3 python3-pip -y
   
   # Install OpenCV dependencies
   sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev -y
   ```

3. **Deploy Application**
   ```bash
   # Clone repository
   git clone <repository-url>
   cd Threat-Detection
   
   # Install Python dependencies
   pip3 install -r requirements.txt
   
   # Run setup
   python3 scripts/setup.py
   ```

4. **Configure Firewall**
   ```bash
   # Allow HTTP traffic
   sudo ufw allow 5000
   sudo ufw enable
   ```

5. **Run Application**
   ```bash
   # Start web interface
   python3 scripts/run_detection.py web
   ```

#### Google Cloud Platform (GCP) Deployment

1. **Create VM Instance**
   ```bash
   gcloud compute instances create threat-detection \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --image-family=debian-11 \
     --image-project=debian-cloud
   ```

2. **Deploy Application**
   ```bash
   # Connect to instance
   gcloud compute ssh threat-detection --zone=us-central1-a
   
   # Install dependencies and deploy (same as AWS)
   ```

#### Azure Deployment

1. **Create Virtual Machine**
   ```bash
   az vm create \
     --resource-group myResourceGroup \
     --name threat-detection \
     --image UbuntuLTS \
     --size Standard_D4s_v3 \
     --admin-username azureuser
   ```

2. **Deploy Application**
   ```bash
   # Connect and deploy (similar to AWS/GCP)
   ```

### Edge Deployment

#### Raspberry Pi Deployment

1. **Install Raspberry Pi OS**
   ```bash
   # Flash Raspberry Pi OS to SD card
   # Boot Raspberry Pi
   ```

2. **Install Dependencies**
   ```bash
   # Update system
   sudo apt-get update && sudo apt-get upgrade -y
   
   # Install Python and dependencies
   sudo apt-get install python3 python3-pip python3-opencv -y
   
   # Install additional dependencies
   sudo apt-get install libatlas-base-dev libhdf5-dev libhdf5-serial-dev -y
   ```

3. **Deploy Application**
   ```bash
   # Clone and setup (same as other platforms)
   # Use lighter models for better performance
   ```

#### NVIDIA Jetson Deployment

1. **Flash Jetson OS**
   ```bash
   # Use NVIDIA SDK Manager to flash Jetson OS
   ```

2. **Install Dependencies**
   ```bash
   # Install Python dependencies
   pip3 install -r requirements.txt
   
   # Use TensorRT optimized models
   ```

3. **Optimize for Jetson**
   ```bash
   # Use TensorRT for model optimization
   # Reduce model complexity for better performance
   ```

## Configuration Management

### Environment Variables

Create `.env` file for configuration:

```env
# Detection Settings
CONFIDENCE_THRESHOLD=0.4
IMAGE_SIZE=640x480
FPS=30

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=5000
WEB_DEBUG=False

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Model Paths
MODEL_DIR=data/models
```

### Configuration Files

#### Production Configuration

```python
# config/production.py
class ProductionConfig:
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = 5000
    CONFIDENCE_THRESHOLD = 0.5
    LOG_LEVEL = 'WARNING'
```

#### Development Configuration

```python
# config/development.py
class DevelopmentConfig:
    DEBUG = True
    HOST = 'localhost'
    PORT = 5000
    CONFIDENCE_THRESHOLD = 0.3
    LOG_LEVEL = 'DEBUG'
```

## Monitoring and Logging

### Application Monitoring

1. **System Metrics**
   ```bash
   # Monitor CPU and memory usage
   htop
   
   # Monitor GPU usage (if available)
   nvidia-smi
   ```

2. **Application Logs**
   ```bash
   # View application logs
   tail -f logs/threat_detection_*.log
   
   # Monitor detection events
   grep "DETECTION" logs/threat_detection_*.log
   ```

3. **Performance Monitoring**
   ```python
   # Use the built-in logger
   from src.utils.logger import logger
   
   logger.info(f"FPS: {current_fps}")
   logger.info(f"Memory usage: {memory_usage}MB")
   logger.info(f"Detection accuracy: {accuracy}%")
   ```

### Health Checks

Create health check endpoint:

```python
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
```

## Security Considerations

### Network Security

1. **Firewall Configuration**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 5000/tcp
   sudo ufw deny 22/tcp  # If not using SSH
   ```

2. **SSL/TLS Configuration**
   ```python
   # Use HTTPS in production
   app.run(ssl_context='adhoc', host='0.0.0.0', port=443)
   ```

### Data Security

1. **Model Security**
   ```bash
   # Secure model files
   chmod 600 data/models/*
   ```

2. **Log Security**
   ```bash
   # Secure log files
   chmod 640 logs/*
   ```

## Scaling and Performance

### Horizontal Scaling

1. **Load Balancer Setup**
   ```bash
   # Use nginx as load balancer
   sudo apt-get install nginx
   ```

2. **Multiple Instances**
   ```bash
   # Run multiple instances on different ports
   python scripts/run_detection.py web --port 5001
   python scripts/run_detection.py web --port 5002
   ```

### Vertical Scaling

1. **GPU Acceleration**
   ```bash
   # Install CUDA and cuDNN
   # Use GPU-optimized models
   ```

2. **Memory Optimization**
   ```python
   # Use model quantization
   # Implement batch processing
   ```

## Backup and Recovery

### Model Backup

```bash
# Backup model files
tar -czf models_backup_$(date +%Y%m%d).tar.gz data/models/

# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/
```

### Data Recovery

```bash
# Restore from backup
tar -xzf models_backup_20231201.tar.gz
tar -xzf config_backup_20231201.tar.gz
```

## Troubleshooting

### Common Issues

1. **Camera Access**
   ```bash
   # Check camera permissions
   ls -la /dev/video*
   
   # Grant permissions
   sudo usermod -a -G video $USER
   ```

2. **Model Loading Errors**
   ```bash
   # Check model file existence
   ls -la data/models/
   
   # Verify model integrity
   python -c "import torch; torch.load('data/models/model.pt')"
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   free -h
   
   # Reduce model complexity
   # Use smaller input sizes
   ```

### Performance Optimization

1. **Reduce Resolution**
   ```python
   # Use smaller input size
   IMAGE_SIZE = (320, 240)
   ```

2. **Model Optimization**
   ```python
   # Use quantized models
   # Enable TensorRT optimization
   ```

3. **Batch Processing**
   ```python
   # Process multiple frames together
   # Use threading for I/O operations
   ``` 