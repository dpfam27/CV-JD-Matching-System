from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from pathlib import Path
import json
import tensorflow as tf
import os

# Explicitly set up all paths
FRONTEND_DIR = Path(__file__).resolve().parent  
PROJECT_ROOT = FRONTEND_DIR.parent.parent       
BACKEND_DIR = PROJECT_ROOT / 'backend' / 'src'  

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

# Initialize Flask app
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Add configuration to disable caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global variable to store current scan result
current_scan_result = None

def load_ml_models():
    """Load and return ML models with error handling"""
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

@app.after_request
def add_header(response):
    """Add headers to prevent caching"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/')
def home():
    """Render home page"""
    return render_template('Aifinalscreen.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file uploads and text input"""
    try:
        resume = request.files['resume'] if 'resume' in request.files else None
        job_desc = request.files['job_description'] if 'job_description' in request.files else None
        
        # Handle text input if files aren't provided
        resume_text = request.form.get('resume_text', '')
        job_desc_text = request.form.get('job_desc_text', '')
        
        # Process the inputs using your ML model
        # For demo, we'll use mock results
        match_score = 85
        skills_match = {
            "JavaScript": (3, 3),
            "React": (2, 3),
            "Node.js": (1, 2)
        }
        missing_keywords = ["TypeScript", "AWS"]
        
        # Store results in global variable for demo
        global current_scan_result
        current_scan_result = {
            "match_score": match_score,
            "skills_match": skills_match,
            "missing_keywords": missing_keywords
        }
        
        return redirect(url_for('result'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result')
def result():
    """Render result page"""
    try:
        return render_template('result.html')
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/history')
def history():
    """Render history page with mock data"""
    try:
        scan_history = [
            {
                "id": 1,
                "score": 85,
                "jobTitle": "Frontend Developer",
                "company": "ABC Company",
                "date": "2025-01-08",
                "ats": "Workday",
                "skills": ["JavaScript", "React", "Node.js"]
            },
            {
                "id": 2,
                "score": 65,
                "jobTitle": "Full Stack Developer",
                "company": "XYZ Corp",
                "date": "2025-01-07",
                "ats": "Greenhouse",
                "skills": ["Python", "Django", "PostgreSQL"]
            }
        ]
        return render_template('History.html', scan_history=scan_history)
    except Exception as e:
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