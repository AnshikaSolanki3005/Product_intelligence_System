"""
Preprocess raw product dataset.
"""
import pandas as pd 
import re 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR/"data"/"raw"/"pricerunner_aggregate.csv"
CLEAN_PATH = BASE_DIR/"data"/"processed"/"products.csv"

def normalize_column_names(df):
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

def clean_text(text):
    text=str(text).lower()
    text=re.sub(r'[^a-z0-9\s]','',text)
    text=re.sub(r'\s+',' ',text).strip()
    return text

def run_preprocess():
    print("Loading raw dataset...")
    df=pd.read_csv(RAW_PATH)
    print("Dataset shape:",df.shape)
    print("Normalizing column names...")
    df=normalize_column_names(df)
    print("Handling missing by values...")
    df=df.fillna("")

    text_columns=["product_title","cluster_label","category_label"]

    print("Cleaning text columns:",text_columns)

    for col in text_columns:
        df[col]=df[col].apply(clean_text)
    print("Creating clean_title column...")
    df["clean_title"] = df["product_title"]

    print("Saving processed dataset...")

    CLEAN_PATH.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(CLEAN_PATH,index=False)
    print("Clean dataset saved:",CLEAN_PATH)
    
if __name__=="__main__":
    run_preprocess()