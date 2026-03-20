"""
clustering.py
-------------
Cluster similar products using KMeans.
"""
import pandas as pd 
import numpy as np 
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from pathlib import Path
import joblib

BASE_DIR= Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR/"data"/"processed"/"products.csv"
EMBED_PATH = BASE_DIR/"models"/"embeddings.npz"
MODEL_PATH = BASE_DIR/"models"/"clustering_model.pkl"
OUTPUT_PATH = BASE_DIR/"models"/"product_catalog.csv"

def run_clustering():
    print("Loading Dataset...")
    df=pd.read_csv(DATA_PATH)

    print("Loading Embedding...")
    embeddings=load_npz(EMBED_PATH)

    print("Embeddding shape:", embeddings.shape)

    if len(df)!=embeddings.shape[0]:
        raise ValueError("Mismatch between dataset rows and embedding")
    
    n_clusters=int(np.sqrt(len(df)))
    print(f"running KMeans Clustering with {n_clusters} clusters...")

    model=KMeans(n_clusters=n_clusters,random_state=42,n_init="auto")
    clusters=model.fit_predict(embeddings)

    df["predicted_cluster"]=clusters

    print("Cluster distribution:")
    print(pd.Series(clusters).value_counts().head())

    print("Saving clustering model...")

    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,MODEL_PATH)

    print("Saving clustered dataset...")

    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUTPUT_PATH,index=False)
    print(f"Results saved to: {OUTPUT_PATH}")
    print("Clustering complete.")

if __name__=="__main__":
    run_clustering()