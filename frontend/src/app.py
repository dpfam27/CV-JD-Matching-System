from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, session
from pathlib import Path
import json
import tensorflow as tf
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

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
app.secret_key = 'your-secret-key-here'  # Required for session

# Add configuration to disable caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

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
        # Get form data
        resume_text = request.form.get('resume_text', '')
        job_desc_text = request.form.get('job_desc_text', '')
        
        if not resume_text or not job_desc_text:
            return jsonify({"error": "Both resume and job description are required"}), 400
        
        # Use the loaded ML model to process the inputs
        # This assumes you have the model and tokenizer loaded globally
        processed_resume = tokenizer.encode(resume_text)
        processed_jd = tokenizer.encode(job_desc_text)
        
        # Get predictions from model
        prediction = model.predict([processed_resume, processed_jd])
        match_score = int(prediction[0] * 100)  # Convert to percentage
        
        # Extract skills (this is a simplified version - you'll need to implement your actual skill extraction logic)
        from sklearn.feature_extraction.text import CountVectorizer
        import numpy as np
        
        # Create a vocabulary of common programming skills
        common_skills = ["javascript", "python", "java", "react", "node.js", "html", "css", "sql", "aws", 
                        "typescript", "angular", "vue", "docker", "kubernetes", "git", "agile"]
        
        vectorizer = CountVectorizer(vocabulary=common_skills)
        
        # Count occurrences in resume and job description
        resume_skills = vectorizer.fit_transform([resume_text.lower()])
        jd_skills = vectorizer.fit_transform([job_desc_text.lower()])
        
        resume_skill_counts = resume_skills.toarray()[0]
        jd_skill_counts = jd_skills.toarray()[0]
        
        # Compare skills
        skills_match = {}
        missing_keywords = []
        for skill, idx in vectorizer.vocabulary_.items():
            resume_count = resume_skill_counts[idx]
            jd_count = jd_skill_counts[idx]
            if jd_count > 0:  # Only include skills mentioned in JD
                skills_match[skill] = (resume_count, jd_count)
                if resume_count == 0:
                    missing_keywords.append(skill)
        
        # Store results in session
        session['scan_result'] = {
            "match_score": match_score,
            "skills_match": skills_match,
            "missing_keywords": missing_keywords,
            "resume_text": resume_text[:100] + "...",  # Store preview for history
            "job_desc_text": job_desc_text[:100] + "..."
        }
        
        # Store in history (you should implement proper database storage here)
        if 'scan_history' not in session:
            session['scan_history'] = []
            
        scan_history = session['scan_history']
        scan_history.append({
            "id": len(scan_history) + 1,
            "score": match_score,
            "jobTitle": "Job from Description",  # You might want to extract this from the JD
            "company": "Company Name",  # You might want to extract this from the JD
            "date": datetime.now().strftime("%Y-%m-%d"),
            "ats": "Generic ATS",
            "skills": list(skills_match.keys())
        })
        session['scan_history'] = scan_history
        
        return redirect(url_for('result'))
    except Exception as e:
        print(f"Error in upload: {str(e)}")  # For debugging
        return jsonify({"error": str(e)}), 500

@app.route('/result')
def result():
    """Render result page"""
    try:
        return render_template('result.html')
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/process_scan', methods=['POST'])
def process_scan():
    """Process the scan data and return results"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        job_desc_text = data.get('job_desc_text', '')
        
        if not resume_text or not job_desc_text:
            return jsonify({"error": "Both resume and job description are required"}), 400
            
        # Process the inputs using your ML model
        # For demo, we'll use mock results
        match_score = 85
        skills_match = {
            "JavaScript": [3, 3],
            "React": [2, 3],
            "Node.js": [1, 2]
        }
        missing_keywords = ["TypeScript", "AWS"]
        
        return jsonify({
            "match_score": match_score,
            "skills_match": skills_match,
            "missing_keywords": missing_keywords
        })
    except Exception as e:
        print(f"Error in process_scan: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/history')
def history():
    """Render history page with actual scan history"""
    try:
        scan_history = session.get('scan_history', [])
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