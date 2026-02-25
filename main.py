from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import engine
import yfinance as yf

app = FastAPI(title="Portfolio Velocity API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BLView(BaseModel):
    type: str  # 'A' or 'R'
    asset: Optional[int] = None
    bull: Optional[int] = None
    bear: Optional[int] = None
    value: float

class AnalysisRequest(BaseModel):
    symbols: List[str]
    views: List[BLView]
    is_auto: Optional[bool] = True
    manual_weights: Optional[dict] = None

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    try:
        # Convert Pydantic models to dicts for the engine
        views_dict = [v.dict() for v in request.views]
        result = engine.run_analysis(
            request.symbols, 
            views_dict, 
            is_auto=request.is_auto, 
            manual_weights=request.manual_weights
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(q: str):
    """
    Validates a ticker symbol with Yahoo Finance.
    Returns ticker and shortName if valid.
    """
    try:
        t = yf.Ticker(q.upper())
        info = t.info
        if "shortName" not in info and "longName" not in info:
            raise HTTPException(status_code=404, detail="Ticker not found on Yahoo Finance")
        
        return {
            "symbol": q.upper(),
            "name": info.get("shortName", info.get("longName", q.upper())),
            "currency": info.get("currency", "USD")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail="Invalid ticker or data unavailable")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
