"""
train_model.py
--------------
Train classification model for product category prediction.
"""
import pandas as pd 
import numpy as np 
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pathlib import Path
import joblib 

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR/"data"/"processed"/"products.csv"
EMBED_PATH = BASE_DIR/"models"/"embeddings.npz"
MODEL_PATH = BASE_DIR/"models"/"classifier.pkl"

def run_training():
    print("Loadoing Dataset...")
    df=pd.read_csv(DATA_PATH)
    print("Loading Embeddings...")
    X=load_npz(EMBED_PATH)
    y=df["category_label"]

    if X.shape[0] != len(y):
        raise ValueError(f"Embeddings rows ({X.shape[0]}) don't match dataset rows ({len(y)}). Re-run feature engineering.")

    print("Splitting the dataset into two...")
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    print("Training classifier...")
    model=LogisticRegression(max_iter=1000,solver="lbfgs")

    model.fit(X_train,y_train)
    print("Evaluating model...")
    preds=model.predict(X_test)

    print("\nClasification Report:\n")
    print(classification_report(y_test,preds))
    print("Saving trained model...")
    
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,MODEL_PATH)
    print(f"Model saved to : {MODEL_PATH}")
    print("Training complete.")

if __name__=="__main__":
    run_training()