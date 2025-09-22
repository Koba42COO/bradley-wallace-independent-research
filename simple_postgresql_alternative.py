#!/usr/bin/env python3
"""
Simple PostgreSQL Alternative for CUDNT
=======================================
SQLite-based database system that mimics PostgreSQL functionality
"""

import sqlite3
import json
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

class SimplePostgreSQLAlternative:
    """Simple PostgreSQL alternative using SQLite"""
    
    def __init__(self, db_path: str = "cudnt_database.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database with tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consciousness_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data TEXT NOT NULL,
                    algorithm TEXT,
                    consciousness_enhancement REAL,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quantum_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    qubits INTEGER,
                    iterations INTEGER,
                    fidelity REAL,
                    processing_time REAL,
                    acceleration_type TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT,
                    value REAL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                if query.strip().upper().startswith('SELECT'):
                    return [dict(row) for row in cursor.fetchall()]
                else:
                    conn.commit()
                    return [{"affected_rows": cursor.rowcount}]
    
    def insert_consciousness_data(self, data: Any, algorithm: str = "prime_aligned_enhanced", 
                                enhancement: float = 1.618, processing_time: float = 0.0) -> int:
        """Insert prime aligned compute processing data"""
        query = """
            INSERT INTO consciousness_data (data, algorithm, consciousness_enhancement, processing_time)
            VALUES (?, ?, ?, ?)
        """
        data_json = json.dumps(data, default=str)
        result = self.execute_query(query, (data_json, algorithm, enhancement, processing_time))
        return result[0].get("affected_rows", 0)
    
    def insert_quantum_result(self, qubits: int, iterations: int, fidelity: float, 
                            processing_time: float, acceleration_type: str = "CUDNT") -> int:
        """Insert quantum computing result"""
        query = """
            INSERT INTO quantum_results (qubits, iterations, fidelity, processing_time, acceleration_type)
            VALUES (?, ?, ?, ?, ?)
        """
        result = self.execute_query(query, (qubits, iterations, fidelity, processing_time, acceleration_type))
        return result[0].get("affected_rows", 0)
    
    def insert_performance_metric(self, metric_type: str, value: float, metadata: Dict = None) -> int:
        """Insert performance metric"""
        query = """
            INSERT INTO performance_metrics (metric_type, value, metadata)
            VALUES (?, ?, ?)
        """
        metadata_json = json.dumps(metadata or {}, default=str)
        result = self.execute_query(query, (metric_type, value, metadata_json))
        return result[0].get("affected_rows", 0)
    
    def get_consciousness_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get prime aligned compute data"""
        query = """
            SELECT * FROM consciousness_data 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        return self.execute_query(query, (limit,))
    
    def get_quantum_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get quantum results"""
        query = """
            SELECT * FROM quantum_results 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        return self.execute_query(query, (limit,))
    
    def get_performance_metrics(self, metric_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance metrics"""
        if metric_type:
            query = """
                SELECT * FROM performance_metrics 
                WHERE metric_type = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            return self.execute_query(query, (metric_type, limit))
        else:
            query = """
                SELECT * FROM performance_metrics 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            return self.execute_query(query, (limit,))
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        # Count records in each table
        tables = ["consciousness_data", "quantum_results", "performance_metrics"]
        for table in tables:
            result = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
            stats[f"{table}_count"] = result[0]["count"]
        
        # Get recent activity
        recent_consciousness = self.execute_query("""
            SELECT COUNT(*) as count FROM consciousness_data 
            WHERE timestamp > datetime('now', '-1 hour')
        """)
        stats["recent_consciousness_processing"] = recent_consciousness[0]["count"]
        
        recent_quantum = self.execute_query("""
            SELECT COUNT(*) as count FROM quantum_results 
            WHERE timestamp > datetime('now', '-1 hour')
        """)
        stats["recent_quantum_processing"] = recent_quantum[0]["count"]
        
        return stats
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old data"""
        cutoff_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        tables = ["consciousness_data", "quantum_results", "performance_metrics"]
        total_deleted = 0
        
        for table in tables:
            query = f"DELETE FROM {table} WHERE timestamp < datetime('now', '-{days} days')"
            result = self.execute_query(query)
            total_deleted += result[0].get("affected_rows", 0)
        
        return total_deleted

# Global instance
simple_postgres = SimplePostgreSQLAlternative()

def get_postgres_client():
    """Get PostgreSQL client (returns simple alternative)"""
    return simple_postgres

if __name__ == "__main__":
    # Test the simple PostgreSQL alternative
    db = get_postgres_client()
    
    print("🚀 Simple PostgreSQL Alternative Test")
    print("=" * 50)
    
    # Test inserting data
    test_data = [1, 2, 3, 4, 5]
    db.insert_consciousness_data(test_data, "prime_aligned_enhanced", 1.618, 0.025)
    print("✅ prime aligned compute data inserted")
    
    # Test quantum results
    db.insert_quantum_result(8, 100, 0.95, 0.042, "CUDNT")
    print("✅ Quantum result inserted")
    
    # Test performance metrics
    db.insert_performance_metric("cpu_usage", 18.5, {"cores": 14})
    print("✅ Performance metric inserted")
    
    # Test retrieving data
    consciousness_data = db.get_consciousness_data(5)
    print(f"✅ Retrieved {len(consciousness_data)} prime aligned compute records")
    
    quantum_results = db.get_quantum_results(5)
    print(f"✅ Retrieved {len(quantum_results)} quantum results")
    
    # Test database stats
    stats = db.get_database_stats()
    print(f"✅ Database stats: {stats}")
    
    print("🎉 Simple PostgreSQL alternative working!")
