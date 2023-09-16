from flask import Flask, request, jsonify
import os
import PyPDF2
from docx import Document
# imports
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
#import numpy as np

app = Flask(__name__)
nlp = spacy.load("en_core_web_md")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher) 

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text


def extract_text_from_folder(folder_path):
    extracted_text_dict = {}  # Initialize an empty dictionary
    
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.lower().endswith('.pdf'):
                extracted_text = extract_text_from_pdf(file_path)
            elif file_name.lower().endswith('.docx'):
                extracted_text = extract_text_from_docx(file_path)
            else:
                extracted_text = ''  # Handle other file types if needed
            
            extracted_text_dict[file_name] = extracted_text  # Store text with filename as key
    
    return extracted_text_dict

# Function to calculate skill match score (cosine similarity)
def calculate_skill_match_score(resume_text, job_description_text):
    # Process the text using spaCy to get document representations
    resume_doc = nlp(resume_text)
    job_description_doc = nlp(job_description_text)

    # Calculate cosine similarity using spaCy's built-in similarity method
    similarity = resume_doc.similarity(job_description_doc)

    return similarity

@app.route('/', methods=['GET'])
def test():
    return "test"

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'folder' not in request.form:
        return jsonify({'error': 'No folder path provided'})

    folder_path = request.form['folder']
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder not found'})

    if 'job_description' not in request.form:
        return jsonify({'error': 'No job description provided'})

    job_description = request.form['job_description']

    extracted_text_dict = extract_text_from_folder(folder_path)

    # Convert job description and its skills to a single text
    job_description_annotations = skill_extractor.annotate(job_description)
    job_description_annotations_list = list({match['doc_node_value'] for match in job_description_annotations['results']['full_matches']})
    job_description_text = " ".join(job_description_annotations_list)

    # Create dictionaries to store skills and scores for each resume
    job_description_skills = job_description_annotations_list
    resumes_skills = {}
    resumes_scores = {}

    for filename, text in extracted_text_dict.items():
        # Extract skills using the SkillExtractor for each file's text
        resumes_annotations = skill_extractor.annotate(text)
        resumes_annotations_list = list({match['doc_node_value'] for match in resumes_annotations['results']['full_matches']})

        # Convert resume skills to a single text
        resume_text = " ".join(resumes_annotations_list)

        # Calculate the skill match score (cosine similarity)
        similarity_score = calculate_skill_match_score(resume_text, job_description_text)
        cosine_similarity_score = round(similarity_score * 100, 2)

        resumes_skills[filename] = resumes_annotations_list
        resumes_scores[filename] = cosine_similarity_score
        
        # Clear resume_text after processing each resume
        resume_text = ""

    result_dict = {
        'job_description_skills': job_description_skills,
        'resumes_skills': resumes_skills,
        'resumes_scores': resumes_scores
    }

    return jsonify(result_dict)


