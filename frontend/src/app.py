from flask import Flask, render_template
from pathlib import Path
import json
import tensorflow as tf
import os

# Explicitly set up all paths
FRONTEND_DIR = Path(__file__).resolve().parent  # src directory
PROJECT_ROOT = FRONTEND_DIR.parent.parent       # CV-JD-Matching-System directory
BACKEND_DIR = PROJECT_ROOT / 'backend' / 'src'  # backend/src directory

# Model paths
MODEL_PATH = BACKEND_DIR / 'models' / 'model.h5'
TOKENIZER_PATH = BACKEND_DIR / 'models' / 'tokenizer.json'
TEMPLATE_DIR = FRONTEND_DIR / 'templates'

# Debug prints for path verification
print("\nDirectory Structure:")
print(f"Frontend Directory: {FRONTEND_DIR}")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Backend Directory: {BACKEND_DIR}")
print(f"Template Directory: {TEMPLATE_DIR}")

print("\nModel Files:")
print(f"Model Path: {MODEL_PATH}")
print(f"Tokenizer Path: {TOKENIZER_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")
print(f"Tokenizer exists: {TOKENIZER_PATH.exists()}")

from flask import Flask, render_template, send_file
app = Flask(__name__, 
    template_folder=str(TEMPLATE_DIR)
)

# Add this configuration to disable caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Initialize Flask app with correct template directory
app = Flask(__name__, 
    template_folder=str(TEMPLATE_DIR)
)

# Load ML models with better error handling
def load_ml_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer file not found at: {TOKENIZER_PATH}")
    
    try:
        print("\nLoading model...")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print("Model loaded successfully")

        print("Loading tokenizer...")
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer = json.load(f)
        print("Tokenizer loaded successfully")
        
        return model, tokenizer
    except Exception as e:
        print(f"\nError during model loading: {str(e)}")
        raise

# Routes
@app.route('/')
def home():
    try:
        return render_template('Aifinalscreen.html')
    except Exception as e:
        print(f"Error rendering home template: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/result')
def result():
    try:
        return render_template('result.html')
    except Exception as e:
        print(f"Error rendering result template: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/history')
def history():
    try:
        return render_template('History.html')
    except Exception as e:
        print(f"Error rendering history template: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    try:
        # Load ML models before starting the server
        model, tokenizer = load_ml_models()
        
        print("\nAll models loaded successfully!")
        print("\nAvailable routes:")
        print("- Home: http://127.0.0.1:5000/")
        print("- Result: http://127.0.0.1:5000/result")
        print("- History: http://127.0.0.1:5000/history")
        
        print("\nStarting Flask server...")
        app.run(debug=True, host='127.0.0.1', port=5000)
    except Exception as e:
        print(f"\nFatal error during startup: {str(e)}")