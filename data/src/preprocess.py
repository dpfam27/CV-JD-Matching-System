import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
import logging
import spacy
import fitz  # PyMuPDF for PDF
import docx  # python-docx for DOCX
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime

class CVDocumentPreprocessor:
    def __init__(self, config_path: Optional[str] = None):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP components
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load configuration or use defaults
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Compile regex patterns
        self.patterns = {
            'years': re.compile(r'(\d+)[\s-]*years?', re.IGNORECASE),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'dates': re.compile(r'((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*[-–]\s*(?:present|current|now)|(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}))', re.IGNORECASE)
        }

    def _default_config(self) -> Dict:
        """Default configuration for skills and field mapping"""
        return {
            'skills': {
                'programming': ['python', 'java', 'javascript', 'c++', 'ruby', 'sql'],
                'data_science': ['machine learning', 'deep learning', 'data analysis', 'statistics'],
                'web_dev': ['html', 'css', 'react', 'angular', 'node.js'],
                'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
                'tools': ['git', 'jira', 'jenkins', 'tableau', 'power bi']
            },
            'fields': {
                'software_development': ['software engineer', 'developer', 'programmer'],
                'data_science': ['data scientist', 'machine learning', 'ai engineer'],
                'business': ['business analyst', 'project manager', 'product manager'],
                'design': ['ui designer', 'ux designer', 'graphic designer']
            },
            'education_levels': {
                'bachelor': ['b.tech', 'b.e.', 'b.sc', 'bachelor'],
                'master': ['m.tech', 'm.e.', 'm.sc', 'master', 'mba'],
                'phd': ['ph.d', 'doctorate', 'doctoral']
            },
            'experience_levels': {
                'entry': (0, 2),
                'mid': (2, 5),
                'senior': (5, 8),
                'expert': (8, float('inf'))
            }
        }

    def _load_config(self, path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._default_config()

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX using python-docx"""
        try:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep some punctuation
        text = re.sub(r'[^a-z0-9.,@\s-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatization
        tokens = word_tokenize(text)
        text = ' '.join([self.lemmatizer.lemmatize(token) for token in tokens 
                        if token not in self.stop_words])
        
        return text

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract and categorize skills"""
        found_skills = {category: [] for category in self.config['skills'].keys()}
        clean_text = self.clean_text(text)
        
        for category, skills in self.config['skills'].items():
            for skill in skills:
                if skill.lower() in clean_text:
                    found_skills[category].append(skill)
        
        return found_skills

    def extract_field(self, text: str) -> List[str]:
        """Extract professional field"""
        fields = []
        clean_text = self.clean_text(text)
        
        for field, keywords in self.config['fields'].items():
            if any(keyword in clean_text for keyword in keywords):
                fields.append(field)
        
        return list(set(fields))

    def extract_experience_level(self, text: str) -> Dict[str, Union[str, float]]:
        """Extract and categorize experience level"""
        # Find all year mentions
        year_matches = self.patterns['years'].findall(text)
        total_years = sum(float(year) for year in year_matches) if year_matches else 0
        
        # Categorize experience level
        level = "unknown"
        for exp_level, (min_years, max_years) in self.config['experience_levels'].items():
            if min_years <= total_years < max_years:
                level = exp_level
                break
        
        return {
            "level": level,
            "years": total_years
        }

    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information"""
        education_info = []
        doc = self.nlp(text)
        
        # Process each sentence
        for sent in doc.sents:
            edu_entry = {}
            
            # Check for education level
            for level, keywords in self.config['education_levels'].items():
                if any(keyword in sent.text.lower() for keyword in keywords):
                    edu_entry['level'] = level
                    break
            
            # Extract organization (potential institution) and dates
            for ent in sent.ents:
                if ent.label_ == 'ORG' and not edu_entry.get('institution'):
                    edu_entry['institution'] = ent.text
                elif ent.label_ == 'DATE' and not edu_entry.get('year'):
                    edu_entry['year'] = ent.text
            
            if edu_entry:
                education_info.append(edu_entry)
        
        return education_info

    def process_document(self, file_path: str) -> Dict:
        """Process a single document (PDF or DOCX)"""
        try:
            # Extract text based on file type
            file_extension = Path(file_path).suffix.lower()
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                text = self.extract_text_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            if not text:
                raise ValueError("No text could be extracted from the document")
            
            # Process extracted text
            processed_data = {
                'filename': os.path.basename(file_path),
                'file_type': file_extension,
                'clean_text': self.clean_text(text),
                'skills': self.extract_skills(text),
                'fields': self.extract_field(text),
                'experience': self.extract_experience_level(text),
                'education': self.extract_education(text)
            }
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            return {
                'filename': os.path.basename(file_path),
                'error': str(e)
            }

    def process_directory(self, directory_path: str) -> pd.DataFrame:
        """Process all documents in a directory"""
        results = []
        valid_extensions = ('.pdf', '.docx', '.doc')
        
        for file_path in Path(directory_path).rglob('*'):
            if file_path.suffix.lower() in valid_extensions:
                result = self.process_document(str(file_path))
                results.append(result)
        
        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save processed results to file"""
        try:
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.json'):
                df.to_json(output_path, orient='records', indent=2)
            elif output_path.endswith('.xlsx'):
                df.to_excel(output_path, index=False)
            else:
                raise ValueError("Unsupported output format. Use .csv, .json, or .xlsx")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CVDocumentPreprocessor()
    
    # Process single document
    cv_path = "data/raw/cv/train"
    result = preprocessor.process_document(cv_path)
    print(f"Processed document: {result['filename']}")
    
    # Process directory
    directory_path = "data/processed/cv"
    results_df = preprocessor.process_directory(directory_path)
    
    # Save results
    preprocessor.save_results(results_df, "cv_analysis_results.xlsx")