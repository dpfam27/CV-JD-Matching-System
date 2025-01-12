# backend/src/services/app.py
import sys
import os
import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import json
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print startup information
print("="*50)
print("Starting CV-JD Matching API")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print("="*50)

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

def preprocess_text(text, tokenizer, maxlen=100):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

@app.post("/compare")
async def compare_cv_jd(cv_file: UploadFile = File(...), jd_file: UploadFile = File(...)):
    try:
        # Read and preprocess CV
        cv_text = fitz.open(stream=cv_file.file.read(), filetype="pdf").load_page(0).get_text("text")
        cv_sequence = preprocess_text(cv_text, tokenizer)

        # Read and preprocess JD
        jd_text = fitz.open(stream=jd_file.file.read(), filetype="pdf").load_page(0).get_text("text")
        jd_sequence = preprocess_text(jd_text, tokenizer)

        # Predict similarity
        similarity = model.predict([cv_sequence, jd_sequence])[0][0]

        return {"similarity": similarity}
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return {"error": str(e)}

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    from tensorflow.keras.losses import MeanSquaredError  # Import the required loss function

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable

# Properly register 'mse' using the correct name
@register_keras_serializable(package="keras.losses", name="mse")
def custom_mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

try:
    print("\nLoading model...")
    model = load_model(MODEL_PATH, custom_objects={"mse": custom_mse})  # Pass 'custom_mse'
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
