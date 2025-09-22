#!/usr/bin/env python3
"""
Simple Redis Alternative for CUDNT
==================================
In-memory caching system that mimics Redis functionality
"""

import time
import json
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

class SimpleRedisAlternative:
    """Simple Redis alternative using in-memory storage"""
    
    def __init__(self, max_size: int = 10000):
        self.data = {}
        self.expiry = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    def ping(self) -> bool:
        """Test connection"""
        return True
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set key-value pair with optional expiration"""
        with self.lock:
            if len(self.data) >= self.max_size:
                self._evict_oldest()
            
            self.data[key] = value
            if ex:
                self.expiry[key] = time.time() + ex
            else:
                self.expiry[key] = None
            
            self.stats["sets"] += 1
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        with self.lock:
            # Check if key exists and not expired
            if key in self.data:
                if self.expiry.get(key) is None or time.time() < self.expiry[key]:
                    self.stats["hits"] += 1
                    return self.data[key]
                else:
                    # Key expired, remove it
                    del self.data[key]
                    del self.expiry[key]
            
            self.stats["misses"] += 1
            return None
    
    def delete(self, *keys: str) -> int:
        """Delete keys"""
        with self.lock:
            deleted = 0
            for key in keys:
                if key in self.data:
                    del self.data[key]
                    if key in self.expiry:
                        del self.expiry[key]
                    deleted += 1
                    self.stats["deletes"] += 1
            return deleted
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        with self.lock:
            if pattern == "*":
                return list(self.data.keys())
            
            # Simple pattern matching
            import fnmatch
            return [key for key in self.data.keys() if fnmatch.fnmatch(key, pattern)]
    
    def info(self) -> Dict[str, Any]:
        """Get cache information"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / max(total_requests, 1)
            
            return {
                "redis_version": "SimpleRedis-1.0.0",
                "used_memory_human": f"{len(self.data)} keys",
                "connected_clients": 1,
                "total_commands_processed": sum(self.stats.values()),
                "keyspace_hits": self.stats["hits"],
                "keyspace_misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "uptime_in_seconds": int(time.time()),
                "keyspace": len(self.data)
            }
    
    def _evict_oldest(self):
        """Evict oldest key when cache is full"""
        if not self.data:
            return
        
        # Find oldest key
        oldest_key = min(self.data.keys(), key=lambda k: self.expiry.get(k, 0))
        del self.data[oldest_key]
        if oldest_key in self.expiry:
            del self.expiry[oldest_key]
    
    def flushall(self) -> bool:
        """Clear all data"""
        with self.lock:
            self.data.clear()
            self.expiry.clear()
            return True

# Global instance
simple_redis = SimpleRedisAlternative()

def get_redis_client():
    """Get Redis client (returns simple alternative)"""
    return simple_redis

if __name__ == "__main__":
    # Test the simple Redis alternative
    redis_client = get_redis_client()
    
    print("🚀 Simple Redis Alternative Test")
    print("=" * 40)
    
    # Test basic operations
    redis_client.set("test_key", "test_value", ex=60)
    value = redis_client.get("test_key")
    print(f"Set/Get test: {value}")
    
    # Test with JSON data
    test_data = {"name": "CUDNT", "version": "1.0.0", "features": ["vectorization", "prime aligned compute"]}
    redis_client.set("cudnt_info", json.dumps(test_data))
    retrieved = redis_client.get("cudnt_info")
    print(f"JSON test: {json.loads(retrieved) if retrieved else None}")
    
    # Test info
    info = redis_client.info()
    print(f"Cache info: {info['keyspace']} keys, {info['hit_rate']:.2%} hit rate")
    
    print("✅ Simple Redis alternative working!")
