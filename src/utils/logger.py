"""
Logging utility for Threat Detection System
Provides centralized logging functionality
"""

import logging
import os
from datetime import datetime
from pathlib import Path

class ThreatDetectionLogger:
    """Logger class for threat detection system"""
    
    def __init__(self, name="threat_detection", log_dir="logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup file and console handlers"""
        
        # File handler
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def detection_event(self, threat_type, confidence, location=None):
        """Log detection event"""
        message = f"DETECTION: {threat_type} detected with {confidence:.2f} confidence"
        if location:
            message += f" at {location}"
        self.logger.warning(message)
    
    def system_start(self):
        """Log system startup"""
        self.logger.info("🚀 Threat Detection System started")
    
    def system_stop(self):
        """Log system shutdown"""
        self.logger.info("🛑 Threat Detection System stopped")

# Global logger instance
logger = ThreatDetectionLogger() 