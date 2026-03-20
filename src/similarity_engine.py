"""
Similarity engine for product search.
"""

import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import joblib 
import re 
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR/"data"/"processed"/"products.csv"
VECTORIZER_PATH = BASE_DIR/"models"/"vectorizer.pkl"
EMBEDDINGS_PATH = BASE_DIR/"models"/"embeddings.npz"

def clean_query(text):
    text=text.lower()
    text=re.sub(r"[^a-z0-9\s]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text

class ProductSearchEngine:
    def __init__(self):
        print("Initializing Product Search Engine...")

        if not DATA_PATH.exists():
            raise FileNotFoundError(f"{DATA_PATH} not found")
        
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"{VECTORIZER_PATH} not found")
        
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"{EMBEDDINGS_PATH} not found")

        print("Loading processed dataset...")
        self.df=pd.read_csv(DATA_PATH)
        print("Loading tf-idf vectorizer...")
        self.vectorizer=joblib.load(VECTORIZER_PATH)
        print("Loading product embeddings...")
        self.embeddings=load_npz(EMBEDDINGS_PATH)

        print("Embeddings shape:",self.embeddings.shape[0])
        print(f"Indexed products: {len(self.df)}")

        if len(self.df)!=self.embeddings.shape[0]:
            raise ValueError(f"Dataset rows({len(self.df)}) and embeddings({self.embeddings.shape[0]}) it doesnt match . rerun feature engineering.")

        print("Product Search Engine initialized successfully.\n")

    def search(self,query,top_k=5):
        print(f"Searching for :'{query}'")
        query=clean_query(query)
        print("Vectorizing query...")
        query_vec=self.vectorizer.transform([query])

        print("Calculating cosine similarity...")
        similarity_score=cosine_similarity(query_vec,self.embeddings)

        # falttening it for the safety
        similarity_score=similarity_score.flatten()

        top_indices=similarity_score.argsort()[::-1][:top_k]
        results=self.df.iloc[top_indices].copy()

        results['similarity_score']=similarity_score[top_indices]
        
        print(f"Search complete. Top{top_k} results returned.\n")
        return results[["product_title","category_label","similarity_score"]]

        exact_match_idx = self.df[self.df["clean_title"]==query].index
        if len(exact_match_idx)>0:
            similarity_score[exact_match_idx[0]]=1.0            

if __name__=="__main__":
    engine=ProductSearchEngine()
    results=engine.search("apple iphone 8 plus 64gb silver")
    print(results)
    