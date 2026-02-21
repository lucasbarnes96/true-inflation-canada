from datetime import datetime, timezone
import sqlite3
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/community", tags=["community"])

DB_PATH = Path("data/contributions.db")

class ContributionBase(BaseModel):
    category: str = Field(..., description="Category of the price (gas, rent, grocery)")
    price: float = Field(..., gt=0, description="The observed price")
    location: Optional[str] = Field(None, description="City or region")

class ContributionCreate(ContributionBase):
    pass

class Contribution(ContributionBase):
    id: int
    timestamp: str
    verified: bool

class CommunityStats(BaseModel):
    category: str
    count: int
    avg_price: float
    min_price: float
    max_price: float

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@router.post("/submit", response_model=Contribution)
async def submit_contribution(contribution: ContributionCreate, request: Request):
    """Submit a new price observation."""
    # Basic validation logic (could be expanded)
    if contribution.category not in ["gas", "rent", "grocery"]:
        raise HTTPException(status_code=400, detail="Invalid category. Must be gas, rent, or grocery.")
    
    # Range checks to prevent trolling
    if contribution.category == "gas" and (contribution.price < 50 or contribution.price > 300):
         raise HTTPException(status_code=400, detail="Gas price out of reasonable range (50-300 c/L).")

    timestamp = datetime.now(timezone.utc).isoformat()
    client_ip = request.client.host if request.client else "unknown"

    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO contributions (timestamp, category, price, location, verified, source_ip) VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, contribution.category, contribution.price, contribution.location, 0, client_ip)
    )
    
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "id": new_id,
        "timestamp": timestamp,
        "category": contribution.category,
        "price": contribution.price,
        "location": contribution.location,
        "verified": False
    }

@router.get("/stats", response_model=List[CommunityStats])
async def get_community_stats():
    """Get aggregated statistics for community contributions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Simple aggregation: avg price by category for today (or all time for MVP)
    # For MVP, let's just do all time to show data quickly. In prod, filter by date.
    cursor.execute("""
        SELECT 
            category, 
            COUNT(*) as count, 
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price
        FROM contributions 
        GROUP BY category
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    stats = []
    for row in rows:
        stats.append({
            "category": row["category"],
            "count": row["count"],
            "avg_price": round(row["avg_price"], 2),
            "min_price": row["min_price"],
            "max_price": row["max_price"]
        })
        
    return stats
