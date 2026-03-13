from pydantic import BaseModel
from typing import List, Optional


class Product(BaseModel):
    name: str
    brand: str
    price_range: str
    link: Optional[str] = None

# This model represents the final response after AI analysis
class AnalysisResponse(BaseModel):
    condition: str
    confidence: float
    description: str
    suggested_products: List[Product]
    status: str = "success"