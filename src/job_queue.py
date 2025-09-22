#!/usr/bin/env python3
"""
SquashPlot Job Queue System
Handles background plotting operations with persistence
"""

import os
import json
import time
import sqlite3
import threading
import traceback
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PlotJob:
    id: str
    farmer_key: str
    pool_key: Optional[str] = None
    contract: Optional[str] = None
    tmp_dir: str = "/tmp/squashplot"
    tmp_dir2: Optional[str] = None
    final_dir: str = "/plots"
    threads: int = 4
    count: int = 1
    k_size: int = 32
    compression: int = 3
    plotter_mode: str = "hybrid"
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    stage: str = "queued"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class JobQueue:
    """Thread-safe job queue with SQLite persistence"""
    
    def __init__(self, db_path: str = "squashplot_jobs.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.workers = {}
        self.max_concurrent = int(os.getenv('MAX_CONCURRENT_PLOTS', 2))
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plot_jobs (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON plot_jobs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON plot_jobs(created_at)
            """)
    
    def add_job(self, job: PlotJob) -> str:
        """Add a new job to the queue"""
        with self.lock:
            job.created_at = datetime.now()
            job.updated_at = datetime.now()
            
            # Serialize job data properly
            job_data = asdict(job)
            # Convert datetime objects to ISO strings
            for field in ['created_at', 'updated_at', 'start_time', 'end_time']:
                if job_data.get(field):
                    job_data[field] = job_data[field].isoformat()
            # Convert enum to string
            if 'status' in job_data:
                job_data['status'] = job_data['status'].value
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO plot_jobs (id, data, status) VALUES (?, ?, ?)",
                    (job.id, json.dumps(job_data), job.status.value)
                )
            
            # Try to start the job if workers are available
            self._try_start_next_job()
            
            return job.id
    
    def get_job(self, job_id: str) -> Optional[PlotJob]:
        """Get a specific job by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data FROM plot_jobs WHERE id = ?", (job_id,)
            )
            row = cursor.fetchone()
            
            if row:
                data = json.loads(row[0])
                # Convert ISO strings back to datetime objects
                for field in ['created_at', 'updated_at', 'start_time', 'end_time']:
                    if data.get(field):
                        try:
                            data[field] = datetime.fromisoformat(data[field])
                        except (ValueError, TypeError):
                            data[field] = None
                # Convert string back to enum
                if 'status' in data:
                    try:
                        data['status'] = JobStatus(data['status'])
                    except ValueError:
                        data['status'] = JobStatus.PENDING
                
                return PlotJob(**data)
            return None
    
    def get_jobs(self, status: Optional[JobStatus] = None, limit: int = 100) -> List[PlotJob]:
        """Get jobs, optionally filtered by status"""
        with sqlite3.connect(self.db_path) as conn:
            if status:
                cursor = conn.execute(
                    "SELECT data FROM plot_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT data FROM plot_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            
            jobs = []
            for row in cursor.fetchall():
                data = json.loads(row[0])
                # Convert ISO strings back to datetime objects
                for field in ['created_at', 'updated_at', 'start_time', 'end_time']:
                    if data.get(field):
                        try:
                            data[field] = datetime.fromisoformat(data[field])
                        except (ValueError, TypeError):
                            data[field] = None
                # Convert string back to enum
                if 'status' in data:
                    try:
                        data['status'] = JobStatus(data['status'])
                    except ValueError:
                        data['status'] = JobStatus.PENDING
                
                jobs.append(PlotJob(**data))
            
            return jobs
    
    def update_job(self, job: PlotJob):
        """Update an existing job"""
        with self.lock:
            job.updated_at = datetime.now()
            
            # Serialize job data properly
            job_data = asdict(job)
            # Convert datetime objects to ISO strings
            for field in ['created_at', 'updated_at', 'start_time', 'end_time']:
                if job_data.get(field):
                    job_data[field] = job_data[field].isoformat()
            # Convert enum to string
            if 'status' in job_data:
                job_data['status'] = job_data['status'].value
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE plot_jobs SET data = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (json.dumps(job_data), job.status.value, job.id)
                )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self.get_job(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        with self.lock:
            # Stop worker if running
            if job_id in self.workers:
                # Signal worker to stop (implementation depends on plotting library)
                pass
            
            job.status = JobStatus.CANCELLED
            job.end_time = datetime.now()
            self.update_job(job)
            
            # Try to start next job
            self._try_start_next_job()
            
            return True
    
    def _try_start_next_job(self):
        """Try to start the next pending job if workers are available"""
        with self.lock:
            if len(self.workers) >= self.max_concurrent:
                return
            
            # Get next pending job
            pending_jobs = self.get_jobs(status=JobStatus.PENDING, limit=1)
            if not pending_jobs:
                return
            
            job = pending_jobs[0]
            job.status = JobStatus.RUNNING
            job.start_time = datetime.now()
            job.stage = "initializing"
            self.update_job(job)
            
            # Start worker thread
            worker = threading.Thread(
                target=self._run_job,
                args=(job,),
                name=f"PlotWorker-{job.id}"
            )
            self.workers[job.id] = worker
            worker.start()
    
    def _run_job(self, job: PlotJob):
        """Run a plotting job in background thread"""
        try:
            # Import plotting logic
            from squashplot_enhanced import SquashPlotEnhanced, PlotConfig
            
            # Create plot configuration
            config = PlotConfig(
                farmer_key=job.farmer_key,
                pool_key=job.pool_key,
                contract=job.contract,
                tmp_dir=job.tmp_dir,
                tmp_dir2=job.tmp_dir2,
                final_dir=job.final_dir,
                threads=job.threads,
                count=job.count,
                k_size=job.k_size,
                compression_level=job.compression
            )
            
            # Initialize plotter
            plotter = SquashPlotEnhanced()
            
            # Progress callback
            def progress_callback(progress: float, stage: str):
                job.progress = progress
                job.stage = stage
                self.update_job(job)
            
            # Run plotting operation
            if job.plotter_mode == "madmax":
                result = plotter.run_madmax_plotting(config, progress_callback)
            elif job.plotter_mode == "bladebit":
                result = plotter.run_bladebit_plotting(config, progress_callback)
            else:  # hybrid
                result = plotter.run_hybrid_plotting(config, progress_callback)
            
            # Mark as completed
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.stage = "completed"
            job.end_time = datetime.now()
            
        except Exception as e:
            # Mark as failed
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.end_time = datetime.now()
            job.stage = "failed"
            
            # Log error
            print(f"Job {job.id} failed: {e}")
            traceback.print_exc()
        
        finally:
            # Update job and clean up
            self.update_job(job)
            
            with self.lock:
                if job.id in self.workers:
                    del self.workers[job.id]
                
                # Try to start next job
                self._try_start_next_job()
    
    def get_queue_stats(self) -> Dict:
        """Get queue statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM plot_jobs 
                GROUP BY status
            """)
            
            stats = {status.value: 0 for status in JobStatus}
            for row in cursor.fetchall():
                stats[row[0]] = row[1]
            
            stats['active_workers'] = len(self.workers)
            stats['max_concurrent'] = self.max_concurrent
            
            return stats
    
    def cleanup_old_jobs(self, days: int = 30):
        """Clean up old completed/failed jobs"""
        cutoff = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                DELETE FROM plot_jobs 
                WHERE status IN ('completed', 'failed', 'cancelled') 
                AND created_at < ?
            """, (cutoff.isoformat(),))
            
            return result.rowcount

# Global job queue instance
job_queue = JobQueue()