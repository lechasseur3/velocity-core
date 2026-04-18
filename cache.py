import os
import json
import hashlib
import time
from pathlib import Path


CACHE_DIR = Path.home() / ".velocity_cache"


def _get_cache_path(key: str) -> Path:
    """Return the cache file path for a given key."""
    return CACHE_DIR / f"{key}.json"


def _hash_params(symbols: list, period: str) -> str:
    """Generate a SHA256 hash from sorted symbols and period."""
    sorted_symbols = sorted(symbols)
    params_str = json.dumps({"symbols": sorted_symbols, "period": period})
    return hashlib.sha256(params_str.encode()).hexdigest()


def cache_get(key: str):
    """
    Retrieve data from cache if exists and not expired.
    Returns None if cache miss or expired.
    """
    cache_path = _get_cache_path(key)
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        
        # Check expiration
        timestamp = cached.get("_timestamp", 0)
        ttl = cached.get("_ttl", 21600)  # Default 6h
        if time.time() - timestamp > ttl:
            # Expired, remove and return None
            cache_path.unlink()
            return None
        
        # Remove metadata fields
        return {k: v for k, v in cached.items() if not k.startswith("_")}
    except (json.JSONDecodeError, IOError):
        cache_path.unlink()
        return None


def cache_set(key: str, data: dict, ttl: int = 21600):
    """
    Store data in cache with TTL.
    Default TTL: 6 hours (21600 seconds).
    """
    cache_path = _get_cache_path(key)
    
    # Create cache directory if needed
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Add metadata
    cached_data = {
        "_timestamp": time.time(),
        "_ttl": ttl,
        **data
    }
    
    with open(cache_path, "w") as f:
        json.dump(cached_data, f)


def cache_cleanup(ttl: int = 21600):
    """
    Clean up expired cache entries.
    Removes all cache files older than TTL (default 6h).
    """
    if not CACHE_DIR.exists():
        return
    
    current_time = time.time()
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            
            timestamp = cached.get("_timestamp", 0)
            file_ttl = cached.get("_ttl", ttl)
            
            if current_time - timestamp > file_ttl:
                cache_file.unlink()
        except (json.JSONDecodeError, IOError):
            # Remove corrupted files
            cache_file.unlink()


def get_portfolio_cache_key(symbols: list, period: str = "5y") -> str:
    """Generate cache key for portfolio data."""
    return _hash_params(symbols, period)
