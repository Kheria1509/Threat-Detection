#!/usr/bin/env python3
"""
Web Interface Example
Demonstrates how to run the web-based threat detection interface
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from web.live_feed import app

def run_web_interface():
    """Run the web-based threat detection interface"""
    
    print("🌐 Starting Web Interface...")
    print("📱 Access the interface at: http://localhost:5000/live_feed")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        # Run the Flask app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    run_web_interface() 