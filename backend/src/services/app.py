# backend/src/services/app.py
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print startup information
print("="*50)
print("Starting CV-JD Matching API")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print("="*50)

# Rest of your imports
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
from pathlib import Path

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get file paths
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
MODEL_PATH = SRC_DIR / "models" / "model.h5"
TOKENIZER_PATH = SRC_DIR / "models" / "tokenizer.json"

print("\nChecking file paths:")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")
print(f"Tokenizer path: {TOKENIZER_PATH}")
print(f"Tokenizer exists: {TOKENIZER_PATH.exists()}")
print("="*50)

# Load model and tokenizer
try:
    print("\nLoading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")

    print("\nLoading tokenizer...")
    with open(TOKENIZER_PATH, 'r') as f:
        tokenizer_config = json.load(f)
        tokenizer = Tokenizer()
        tokenizer.__dict__.update(json.loads(tokenizer_config))
    print("Tokenizer loaded successfully!")
    print("="*50)
except Exception as e:
    print(f"\nERROR loading model or tokenizer:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("="*50)
    raise

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.get("/")
async def root():
    return {"message": "CV-JD Matching API is running"}

if __name__ == "__main__":
    import uvicorn
    print("Starting uvicorn server...")
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    import uvicorn
    print("\nStarting uvicorn server...")
    print("API will be available at http://localhost:8000")
    print("Documentation will be available at http://localhost:8000/docs")
    print("="*50)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)