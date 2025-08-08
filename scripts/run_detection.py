#!/usr/bin/env python3
"""
Main script to run the Threat Detection System
Usage: python scripts/run_detection.py [mode]
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_detection.py [mode]")
        print("Available modes:")
        print("  basic     - Basic fire/smoke detection")
        print("  violence  - Violence detection")
        print("  lstm      - LSTM-based video classification")
        print("  all       - Multi-model detection")
        print("  web       - Web-based interface")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "basic":
        from core.detect import main as run_basic
        run_basic()
    elif mode == "violence":
        from core.vio import print_results
        print_results("./data/models/modelnew.h5")
    elif mode == "lstm":
        from models.lstm import main as run_lstm
        run_lstm()
    elif mode == "all":
        from core.all import main as run_all
        run_all()
    elif mode == "web":
        from web.live_feed import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: basic, violence, lstm, all, web")

if __name__ == "__main__":
    main() 