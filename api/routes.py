"""
routes.py
---------
All API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from api.schemas import SearchResponse, PredictRequest, PredictResponse, ClusterResponse, AnalyticsResponse
from api import services

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1, description="Product search query"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results")
):
    results = services.search_products(query, top_k)
    return SearchResponse(query=query, total=len(results), results=results)


@router.post("/predict")
def predict(request: PredictRequest):
    result = services.predict_category(request.title)
    return result


@router.get("/cluster/{cluster_id}", response_model=ClusterResponse)
def get_cluster(cluster_id: int,top_k: int = Query(default=20, ge=1, le=100)):
    products = services.get_cluster_products(cluster_id, top_k)    
    if not products:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found or empty")
    return ClusterResponse(cluster_id=cluster_id, total=len(products), products=products)


@router.get("/analytics", response_model=AnalyticsResponse)
def analytics():
    return services.get_analytics()