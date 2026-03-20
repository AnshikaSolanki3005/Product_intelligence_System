"""
schemas.py
----------
Request and response shapes for the API.
"""

from pydantic import BaseModel
from typing import List


class SearchResult(BaseModel):
    product_title: str
    category_label: str
    similarity_score: float


class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[SearchResult]


class PredictRequest(BaseModel):
    title: str


class PredictResponse(BaseModel):
    title: str
    predicted_category: str


class ClusterProduct(BaseModel):
    product_title: str
    category_label: str


class ClusterResponse(BaseModel):
    cluster_id: int
    total: int
    products: List[ClusterProduct]


class AnalyticsResponse(BaseModel):
    total_products: int
    total_clusters: int
    category_distribution: dict