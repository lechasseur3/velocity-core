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
    cov_method: Optional[str] = "sample_cov"  # 'sample_cov' | 'ledoit_wolf' | 'oracle_approximating'
    tau: Optional[float] = 0.05
    min_weight: Optional[float] = 0.02
    max_weight: Optional[float] = 0.25
    benchmark: Optional[str] = "SPY"
    risk_free_rate: Optional[float] = None

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    try:
        # Convert Pydantic models to dicts for the engine
        views_dict = [v.dict() for v in request.views]
        result = engine.run_analysis(
            request.symbols, 
            views_dict, 
            is_auto=request.is_auto, 
            manual_weights=request.manual_weights,
            cov_method=request.cov_method,
            tau=request.tau,
            min_weight=request.min_weight,
            max_weight=request.max_weight,
            benchmark=request.benchmark,
            risk_free_rate=request.risk_free_rate
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quotes")
async def quotes(symbols: str):
    """
    Real-time quotes for comma-separated tickers.
    Returns {ticker: {price, change_pct, currency, name}}.
    """
    result = {}
    for t in symbols.upper().split(","):
        t = t.strip()
        if not t:
            continue
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            price = info.get("currentPrice", info.get("regularMarketPrice", info.get("previousClose", 0)))
            change = info.get("regularMarketChangePercent", 0) or 0
            result[t] = {
                "price": price,
                "change_pct": round(float(change), 2),
                "currency": info.get("currency", "USD"),
                "name": info.get("shortName", info.get("longName", t))
            }
        except:
            result[t] = {"price": 0, "change_pct": 0, "currency": "USD", "name": t}
    return result

@app.get("/optimal-portfolio")
async def optimal_portfolio(region: str = "US"):
    """
    Generate a diversified optimal portfolio based on region.
    Regions: US, EU, FR, ASIA, GLOBAL
    """
    portfolios = {
        "US": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "UNH"],
        "EU": ["SAP.DE", "ASML.AS", "NESN.SW", "NOVN.SW", "MC.PA", "OR.PA", "SIE.DE", "SAN.PA"],
        "FR": ["MC.PA", "OR.PA", "SAN.PA", "CAP.PA", "AI.PA", "RMS.PA", "BNP.PA", "ACA.PA"],
        "ASIA": ["7203.T", "9984.T", "0005.HK", "2330.TW", "005930.KS", "BABA", "SE", "INFY"],
        "GLOBAL": ["AAPL", "MSFT", "NVDA", "SAP.DE", "ASML.AS", "MC.PA", "7203.T", "2330.TW"]
    }
    tickers = portfolios.get(region.upper(), portfolios["US"])
    benchmarks = {"US": "SPY", "EU": "EZU", "FR": "EWQ", "ASIA": "AIA", "GLOBAL": "ACWI"}
    bm = benchmarks.get(region.upper(), "SPY")
    try:
        views_dict = []
        result = engine.run_analysis(tickers, views_dict, is_auto=True, cov_method="sample_cov", benchmark=bm)
        return {"tickers": tickers, "benchmark": bm, "region": region, "result": result}
    except Exception as e:
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
