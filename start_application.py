"""
Startup script to start both frontend and backend components of the Internship Matching System
"""

import subprocess
import sys
import os
import time
import webbrowser

def start_backend():
    """Start the backend Flask server"""
    try:
        print("🚀 Starting backend server...")
        # Change to backend directory and run the server
        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        backend_process = subprocess.Popen(
            [sys.executable, "api_server.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Capture both stdout and stderr
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        print("✅ Backend server started successfully")
        return backend_process
    except Exception as e:
        print(f"❌ Failed to start backend server: {e}")
        return None

def open_frontend():
    """Open the frontend in the default web browser"""
    try:
        # Get the absolute path to the index.html file
        frontend_path = os.path.abspath("frontend/index.html")
        frontend_url = f"file://{frontend_path}"
        
        print("🌐 Opening frontend in browser...")
        webbrowser.open(frontend_url)
        print("✅ Frontend opened successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to open frontend: {e}")
        return False

def main():
    """Main function to start both frontend and backend"""
    print("=" * 60)
    print("INTERNSHIP MATCHING SYSTEM - STARTUP SCRIPT")
    print("=" * 60)
    
    # Start backend server
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start application")
        return
    
    # Wait a moment for the server to start
    print("⏳ Waiting for server to initialize...")
    time.sleep(5)
    
    # Open frontend
    frontend_opened = open_frontend()
    
    if frontend_opened:
        print("\n" + "=" * 60)
        print("✅ APPLICATION STARTED SUCCESSFULLY!")
        print("=" * 60)
        print("Backend server is running at: http://localhost:5000")
        print("Frontend is open in your browser")
        print("Navigate to 'AI Recommendations' to get personalized internship matches")
        print("\nPress Ctrl+C to stop the backend server")
        print("=" * 60)
    else:
        print("\n⚠️  Frontend failed to open, but backend is running")
        print("You can manually open frontend/index.html in your browser")
        print("Backend server is running at: http://localhost:5000")
    
    # Keep the backend running and display its output
    try:
        print("\n📝 Backend server output:")
        print("-" * 60)
        while True:
            output = backend_process.stdout.readline()
            if output:
                print(output.strip())
            if backend_process.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping backend server...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Backend server stopped")

if __name__ == "__main__":
    main()