"""
services.py
-----------
Business logic layer between routes and models.
"""
from api.model_loader import models
from src.utils import clean_text, get_logger

logger = get_logger("services")

def search_products(query: str, top_k: int = 10) -> list:
    results = models.search_engine.search(query, top_k=top_k)
    return results.to_dict(orient="records")

def predict_category(title: str):
    try:
        cleaned = clean_text(title)
        vec = models.search_engine.vectorizer.transform([cleaned])

        if hasattr(models.classifier, "predict_proba"):
            probs = models.classifier.predict_proba(vec)[0]
            classes = models.classifier.classes_

            # Getting the BEST category only
            best_idx = probs.argmax()
            predictions = [
                {
                    "category": classes[best_idx],
                    "confidence": float(probs[best_idx])
                }
            ]
        else:
            pred = models.classifier.predict(vec)[0]
            predictions = [
                {"category": pred, "confidence": 1.0}
            ]
        return {
            "query": title,
            "predictions": predictions
        }
    except Exception as e:
        print("Error in predicting product category:", str(e))
        raise e

def get_cluster_products(cluster_id: int, top_k: int = 20) -> list:
    df = models.df
    if "predicted_cluster" not in df.columns:
        return []
    cluster_df = df[df["predicted_cluster"] == cluster_id].head(top_k)
    return cluster_df[["product_title", "category_label"]].to_dict(orient="records")

def get_analytics() -> dict:
    df = models.df
    category_dist = df["category_label"].value_counts().to_dict()
    total_clusters = int(df["predicted_cluster"].nunique()) if "predicted_cluster" in df.columns else 0
    return {
        "total_products": len(df),
        "total_clusters": total_clusters,
        "category_distribution": category_dist,
    }