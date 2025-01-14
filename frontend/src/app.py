from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, session
from pathlib import Path
import json
import tensorflow as tf
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import requests
import base64
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Initialize Udemy Course Recommender
class UdemyCourseRecommender:
    def __init__(self, test_mode: bool = True):
        self.test_mode = test_mode
        self.base_url = 'https://www.udemy.com/api-2.0/'
        
        # Mock data for testing
        self.mock_data = {
            'python': [
                {'title': 'Python Bootcamp 2024', 'url': '/python-bootcamp', 'rating': 4.8, 'price': '$13.99'},
                {'title': 'Python for Data Science', 'url': '/python-data', 'rating': 4.6, 'price': '$12.99'}
            ],
            'javascript': [
                {'title': 'Modern JavaScript', 'url': '/modern-js', 'rating': 4.7, 'price': '$11.99'},
                {'title': 'Full Stack JavaScript', 'url': '/fullstack-js', 'rating': 4.5, 'price': '$14.99'}
            ],
            'react': [
                {'title': 'React Complete Guide', 'url': '/react-guide', 'rating': 4.9, 'price': '$15.99'},
                {'title': 'React and Redux', 'url': '/react-redux', 'rating': 4.7, 'price': '$14.99'}
            ],
            'node.js': [
                {'title': 'Node.js Developer Course', 'url': '/nodejs-dev', 'rating': 4.8, 'price': '$12.99'},
                {'title': 'Complete Node.js Guide', 'url': '/nodejs-complete', 'rating': 4.6, 'price': '$13.99'}
            ]
        }

        if not test_mode:
            self.client_id = os.getenv('UDEMY_CLIENT_ID')
            self.client_secret = os.getenv('UDEMY_CLIENT_SECRET')
            if not all([self.client_id, self.client_secret]):
                print("Warning: Missing API credentials. Switching to test mode.")
                self.test_mode = True
            else:
                self.headers = self._get_headers()

    def _get_headers(self) -> Dict:
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {
            'Authorization': f'Bearer {encoded}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def search_courses(self, skill: str, limit: int = 3) -> List[Dict]:
        if self.test_mode:
            return self.mock_data.get(skill.lower(), [])[:limit]
        
        try:
            endpoint = f"{self.base_url}courses/"
            params = {
                'search': skill,
                'page_size': limit,
                'ordering': 'highest-rated',
                'ratings_gte': 4.0,
                'fields[course]': 'title,url,price,rating'
            }
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get('results', [])
        except Exception as e:
            print(f"Error fetching courses: {e}")
            return self.mock_data.get(skill.lower(), [])[:limit]

    def get_recommendations(self, skills: List[str], limit: int = 5) -> List[Dict]:
        all_courses = []
        for skill in skills:
            courses = self.search_courses(skill)
            formatted = [{
                'title': course['title'],
                'url': course['url'],
                'skill': skill,
                'rating': course.get('rating', 'N/A'),
                'price': course.get('price', 'N/A')
            } for course in courses]
            all_courses.extend(formatted)
        return all_courses[:limit]

# Initialize the course recommender
course_recommender = UdemyCourseRecommender(test_mode=True)

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
        # Initialize with default values
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
    
@app.route('/process_scan', methods=['POST'])
def process_scan():
    """Process the scan data and return results with course recommendations"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '').lower()
        job_desc_text = data.get('job_desc_text', '').lower()
        
        if not resume_text or not job_desc_text:
            return jsonify({"error": "Both resume and job description are required"}), 400

        # Process using the actual ML model
        common_skills = ["python", "java", "javascript", "react", "node.js", "html", "css", 
                        "sql", "aws", "docker", "kubernetes", "git", "agile", "typescript",
                        "angular", "vue", "postgresql", "mongodb", "rest api", "graphql"]

        # Count occurrences in both texts
        vectorizer = CountVectorizer(vocabulary=common_skills)
        resume_skills = vectorizer.fit_transform([resume_text])
        jd_skills = vectorizer.fit_transform([job_desc_text])
        
        resume_skill_counts = resume_skills.toarray()[0]
        jd_skill_counts = jd_skills.toarray()[0]

        # Compare skills
        skills_match = {}
        missing_keywords = []
        for skill, idx in vectorizer.vocabulary_.items():
            resume_count = resume_skill_counts[idx]
            jd_count = jd_skill_counts[idx]
            if jd_count > 0:  # Only include skills mentioned in JD
                skills_match[skill] = [resume_count, jd_count]
                if resume_count == 0:
                    missing_keywords.append(skill)

        # Calculate match score
        total_required = sum(1 for counts in skills_match.values() if counts[1] > 0)
        matched = sum(1 for counts in skills_match.values() if counts[0] >= counts[1])
        match_score = int((matched / total_required * 100) if total_required > 0 else 0)

        # Get course recommendations for missing skills
        course_recommendations = course_recommender.get_recommendations(missing_keywords)

        # Store results
        scan_result = {
            "match_score": match_score,
            "skills_match": skills_match,
            "missing_keywords": missing_keywords,
            "course_recommendations": course_recommendations,
            "date": datetime.now().strftime("%Y-%m-%d")
        }

        # Store in history
        if 'scan_history' not in session:
            session['scan_history'] = []
        
        session['scan_history'].append({
            "id": len(session['scan_history']) + 1,
            "score": match_score,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "skills": list(skills_match.keys())
        })

        return jsonify(scan_result)
    except Exception as e:
        print(f"Error in process_scan: {str(e)}")
        return jsonify({"error": str(e)}), 500

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