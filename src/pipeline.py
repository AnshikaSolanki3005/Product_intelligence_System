import time

from src.preprocess import run_preprocess
from src.feature_engineering import run_features
from src.train_model import run_training
from src.similarity_engine import ProductSearchEngine
from src.clustering import run_clustering

def run_pipeline():
    print("\nStarting ML Pipeline")
    start = time.time()
    print("Step 1 - Preprocessing")
    run_preprocess()

    print("Step 2 - Feature Engineering")
    run_features()

    print("\nStep 3 — Training Classifier")
    run_training()

    print("\nStep 4 — Similarity Search Demo")
    
    engine =ProductSearchEngine()
    results=engine.search("e.g iphone ")

    print(results)

    print("\nStep 5 — Clustering")
    run_clustering()

    print("\nPipeline completed successfully")
    print("Total runtime:", round(time.time() - start, 2), "seconds")

if __name__ == "__main__":
    run_pipeline()