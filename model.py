import os
import pandas as pd
import numpy as np
import timeit
import time 
import datetime 
import json
import pickle
import re
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import tensorflow as tf
import tensorflow_hub as hub


def get_use_embeddings(use_model, texts):
    return use_model(texts).numpy()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove newlines and extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Process with spaCy
    doc = nlp(text)
    
    # Remove stopwords, punctuation, and perform lemmatization
    tokens = [token.lemma_.strip() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha and token.lemma_ != "-PRON-"]
    
    # Join tokens back into a string
    return " ".join(tokens)


def inputs_recommendations(X, user_inputs):
    
    similarities = cosine_similarity(user_inputs, X)
    
    # Get indices of top N similar courses
    N = 20  # Number of recommendations
    top_indices = similarities.argsort()[0][-N:][::-1]
    # Get the recommended courses
    recommended_courses = udemy_courses.iloc[top_indices]
    
    recommended_courses["similarity_score"] = similarities[0][top_indices]
    
    # Select columns to return to user and sort by cosine similarity
    recommended_courses = (
        recommended_courses
        .groupby(["similarity_score", "title", "visible_instructors_0_title", "description", "headline", "course_categories", "instructor_categories", "instructional_level_simple", "last_updated", "duration"], as_index=False)
        .agg(
            avg_rating = ("avg_rating", "max"),
            num_reviews = ("num_reviews", "max"),
            last_update_date = ("last_update_date", "max"),
            is_paid = ("is_paid", "max"),
            has_certificate = ("has_certificate", "max"),
            quizzes = ("quizzes", "max"),
            assignments = ("assignments", "max"),
            bestseller = ("bestseller", "max"),
        )
        .sort_values("similarity_score", ascending=False)
        .reset_index(drop=True)
    )
    
    return recommended_courses

    

if __name__ == "__main__":
    # Load in cleaned data
    udemy_courses = pd.read_csv("udemy_courses_features.csv")
    X_tabular = np.load("x_tabular.npy")

    with open("preprocessor.b", "rb") as f_in:
        preprocessor = pickle.load(f_in)

    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # Load spaCy model
    nlp = spacy.load("en_core_web_lg")  
    
    # Get text embeddings for courses
    saved_course_embeddings = "course_embeddings.npy"
    if os.path.exists(saved_course_embeddings):
        text_embeddings = np.load(saved_course_embeddings)
    else:
        text_embeddings = get_use_embeddings(use_model, udemy_courses['combined_text'])
        np.save("course_embeddings.npy", text_embeddings)

    # TODO: Create a frontend for users to input this information

    # User provides information on desired course
    user_text = """
    I'm interested in learning NLP techniques using Python. I'd like a course that covers text preprocessing, sentiment analysis, and topic modeling.
    """

    user_text_preprocessed = preprocess_text(user_text)
    user_input = get_use_embeddings(use_model, [user_text_preprocessed])

    # Get course recommendations
    recommended_courses = inputs_recommendations(text_embeddings, user_input)

    # TODO: Return output to a frontend then use parameters to filter
    # Save output to CSV, returning only 20 outputs for now
    recommended_courses.to_csv("recommended_courses.csv", index=False)
    
    









