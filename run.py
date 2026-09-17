import sys
import subprocess
import time
import argparse
from pathlib import Path

# Fix Windows console UTF-8 encoding for emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def start_backend():
    print("🚀 Starting FastAPI backend on http://127.0.0.1:8000 ...")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=str(Path(__file__).parent)
    )

def start_frontend():
    print("✨ Starting Streamlit frontend on http://127.0.0.1:8501 ...")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.headless", "true"],
        cwd=str(Path(__file__).parent)
    )

def main():
    parser = argparse.ArgumentParser(description="AI Interview Analyzer Dual Runner")
    parser.add_argument("--mode", choices=["all", "backend", "frontend"], default="all", help="Service to start")
    args = parser.parse_args()

    processes = []
    try:
        if args.mode in ["all", "backend"]:
            p_back = start_backend()
            processes.append(p_back)
            time.sleep(1.5)

        if args.mode in ["all", "frontend"]:
            p_front = start_frontend()
            processes.append(p_front)

        print("\n" + "=" * 60)
        print("  AI INTERVIEW ANALYZER IS RUNNING!")
        print("  - Backend API:    http://127.0.0.1:8000/docs")
        print("  - Streamlit App:  http://127.0.0.1:8501")
        print("=" * 60 + "\n")
        print("Press Ctrl+C to terminate services.")

        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
