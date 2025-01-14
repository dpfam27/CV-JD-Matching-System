from flask import Flask, request, render_template, redirect, url_for, jsonify, session
import os
import numpy as np
from werkzeug.utils import secure_filename
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session

# Configure paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'model.h5'
TOKENIZER_PATH = BASE_DIR / 'models' / 'tokenizer.json'

# Configure uploads
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load tokenizer
try:
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer = json.load(f)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

# Load the model
try:
    model = load_model(str(MODEL_PATH))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def extract_skills(text):
    """Extract skills from text using regex pattern matching"""
    common_skills = ["python", "java", "javascript", "react", "node.js", "html", "css", 
                    "sql", "aws", "docker", "kubernetes", "git", "agile", "typescript",
                    "angular", "vue", "postgresql", "mongodb", "rest api", "graphql"]
    
    # Convert text to lowercase and find matches
    text = text.lower()
    found_skills = []
    for skill in common_skills:
        if skill in text:
            found_skills.append(skill)
    
    return found_skills

def process_text(resume_text, job_text):
    """Process resume and job description text to get match analysis"""
    
    # Get skills from both texts
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    
    # Find missing skills
    missing_keywords = list(set(job_skills) - set(resume_skills))
    
    # Calculate skill match percentage
    required_skills = set(job_skills)
    matched_skills = required_skills.intersection(set(resume_skills))
    skill_match_score = len(matched_skills) / len(required_skills) * 100 if required_skills else 0
    
    # Create skills match dictionary
    skills_match = {}
    for skill in job_skills:
        resume_count = resume_text.lower().count(skill)
        job_count = job_text.lower().count(skill)
        skills_match[skill] = [resume_count, job_count]
    
    return {
        'match_score': round(skill_match_score, 2),
        'skills_match': skills_match,
        'missing_keywords': missing_keywords
    }

@app.route('/')
def home():
    return render_template('Aifinalscreen.html')

@app.route('/result')
def result():
    """Render result page with initial values"""
    try:
        initial_data = {
            "match_score": 0,
            "skills_match": {},
            "missing_keywords": []
        }
        return render_template('result.html', **initial_data)
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
            
        # Process texts and get results
        results = process_text(resume_text, job_desc_text)
        
        # Update history
        if 'scan_history' not in session:
            session['scan_history'] = []
        
        session['scan_history'].append({
            "id": len(session['scan_history']) + 1,
            "score": results['match_score'],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "skills": list(results['skills_match'].keys())
        })

        return jsonify(results)
    except Exception as e:
        print(f"Error in process_scan: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/history')
def history():
    """Render history page"""
    try:
        scan_history = session.get('scan_history', [])
        return render_template('History.html', scan_history=scan_history)
    except Exception as e:
        return f"Error: {str(e)}", 500
    
@app.route('/predict', methods=['POST'])
def predict():
    # Handle form data
    return "Prediction done"

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)