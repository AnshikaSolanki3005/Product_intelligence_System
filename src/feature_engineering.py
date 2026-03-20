"""
Feature engineering module.
Convert product titles into numerical representations using TF-IDF vectorization.
"""
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from pathlib import Path
import joblib

BASE_DIR=Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR/"data"/"processed"/"products.csv"
VECTORIZER_PATH = BASE_DIR/"models"/"vectorizer.pkl"
EMBEDDINGS_PATH = BASE_DIR/"models"/"embeddings.npz"

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataste not found at {DATA_PATH}")
    df= pd.read_csv(DATA_PATH)
    if "clean_title" not in df.columns:
        raise ValueError("Column 'clean_title' not found in dataset")
    
    df.dropna(subset=['clean_title'])
    print("Dataset Shape:",df.shape)
    return df

def create_vectorizer():
    vectorizer=TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        stop_words="english",
        min_df=2,
        strip_accents="unicode"
    )
    return vectorizer

def generate_embeddings(vectorizer,texts):
    embeddings=vectorizer.fit_transform(texts)
    print("Embedding Shape:",embeddings.shape)
    print("Number of features:",len(vectorizer.get_feature_names_out()))
    return embeddings

def save_artifacts(vectorizer,embeddings):
    VECTORIZER_PATH.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(vectorizer,VECTORIZER_PATH)
    save_npz(EMBEDDINGS_PATH,embeddings)
    print("Artifacts saved.")

def run_features():
    print("Loading dataset...")
    df=load_data()
    print("Creating TF-IDF features...")
    vectorizer=create_vectorizer()
    embeddings=generate_embeddings(vectorizer,df["clean_title"])
    print("Saving feature artifacts...")
    save_artifacts(vectorizer,embeddings)
    print("Feature Engineeering complete. ")

if __name__=="__main__":
    run_features()