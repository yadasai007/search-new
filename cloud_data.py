import numpy as np
from sentence_transformers import SentenceTransformer
from extract import *
from datetime import datetime
from aws import AWS

aws=AWS()
# Load the SBERT model (you can choose any available model)
model = SentenceTransformer('all-MiniLM-L6-v2')  # A small but powerful SBERT model

def convert_using_sbert(text):
    # Generate embeddings using SBERT
    embeddings = model.encode(text)
    return np.array(embeddings)  # Convert embeddings to NumPy array

def convert(text):
    # Convert text to SBERT embeddings
    sbert_vector = convert_using_sbert(text)
    return sbert_vector

def search_value(search_key):
    encoded_vector = {}
    pdf_files_text = aws.files_content
    for file in pdf_files_text.keys():
        encoded_vector[file]=convert(pdf_files_text[file])

    search_query=search_key
    search_vector=convert(search_query)
    scores={}
    for vector in encoded_vector.keys():
        score=np.dot(encoded_vector[vector],search_vector)
        scores[vector]=score
    scores=sorted(scores.items(),key=lambda item:item[1],reverse=True)
    scores=scores if len(scores)<=5 else scores[0:5]
    scores=dict(scores)
    for file in scores.keys():
        score=scores[file]
        content=pdf_files_text[file]
        scores[file]=[score,content]
    return scores
