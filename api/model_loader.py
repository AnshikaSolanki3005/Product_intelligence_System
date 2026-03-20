"""
model_loader.py
---------------
Loads all models once when the server starts.
Prevents reloading on every request.
"""

import joblib
import pandas as pd
from pathlib import Path
from src.similarity_engine import ProductSearchEngine
from src.utils import get_logger

logger = get_logger("model_loader")

BASE_DIR = Path(__file__).resolve().parents[1]

CLASSIFIER_PATH = BASE_DIR / "models" / "classifier.pkl"
CATALOG_PATH    = BASE_DIR / "outputs" / "product_catalog.csv"
PRODUCTS_PATH   = BASE_DIR / "data" / "processed" / "products.csv"


class ModelLoader:

    def __init__(self):

        logger.info("Loading search engine...")
        self.search_engine = ProductSearchEngine()

        logger.info("Loading classifier...")
        self.classifier = joblib.load(CLASSIFIER_PATH)

        # use clustered catalog if available, else fallback
        if CATALOG_PATH.exists():
            logger.info("Loading product catalog (clustered)...")
            self.df = pd.read_csv(CATALOG_PATH)
        else:
            logger.info("Clustered catalog not found, loading processed products...")
            self.df = pd.read_csv(PRODUCTS_PATH)

        logger.info(f"Models ready. Products indexed: {len(self.df)}")


models = ModelLoader()