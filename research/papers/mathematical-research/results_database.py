#!/usr/bin/env python3
"""
Wallace Transform Results Database
==================================

Comprehensive database system for storing and retrieving harmonic analysis results.
Supports billion-scale analysis with efficient querying and visualization.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime
import os
from pathlib import Path

class WallaceResultsDatabase:
    def __init__(self, db_path="wallace_results.db"):
        """Initialize the results database."""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Analysis runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    scale INTEGER NOT NULL,
                    primes_count INTEGER NOT NULL,
                    gaps_count INTEGER NOT NULL,
                    fft_sample_size INTEGER,
                    autocorr_sample_size INTEGER,
                    processing_time REAL,
                    checkpoint_file TEXT,
                    status TEXT DEFAULT 'completed'
                )
            ''')

            # Harmonic ratios table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS harmonic_ratios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    ratio_symbol TEXT NOT NULL,
                    ratio_value REAL NOT NULL,
                    method TEXT NOT NULL,
                    detected BOOLEAN DEFAULT FALSE,
                    distance REAL,
                    magnitude REAL,
                    correlation REAL,
                    peak_rank INTEGER,
                    frequency REAL,
                    lag INTEGER,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            ''')

            # Scale analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scale_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    scale INTEGER NOT NULL,
                    primes_at_scale INTEGER,
                    gaps_at_scale INTEGER,
                    gap_mean REAL,
                    gap_std REAL,
                    gap_min INTEGER,
                    gap_max INTEGER,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            ''')

            # Detection statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    total_ratios INTEGER,
                    fft_detections INTEGER,
                    autocorr_detections INTEGER,
                    consensus_detections INTEGER,
                    unique_ratios_detected INTEGER,
                    detection_rate REAL,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            ''')

            # Raw results storage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    result_type TEXT NOT NULL,
                    result_data TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            ''')

            conn.commit()

    def store_analysis_results(self, results, analysis_type='billion_scale', processing_time=None):
        """Store complete analysis results in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert analysis run
            run_data = {
                'timestamp': results.get('metadata', {}).get('analysis_timestamp', datetime.now().isoformat()),
                'analysis_type': analysis_type,
                'scale': results.get('metadata', {}).get('primes_count', 0),
                'primes_count': results.get('metadata', {}).get('primes_count', 0),
                'gaps_count': results.get('metadata', {}).get('total_chunks', 0) * 500000,  # Estimate
                'fft_sample_size': results.get('metadata', {}).get('fft_sample_size'),
                'autocorr_sample_size': results.get('metadata', {}).get('autocorr_sample_size'),
                'processing_time': processing_time,
                'checkpoint_file': results.get('metadata', {}).get('checkpoint_file'),
                'status': 'completed'
            }

            cursor.execute('''
                INSERT INTO analysis_runs
                (timestamp, analysis_type, scale, primes_count, gaps_count,
                 fft_sample_size, autocorr_sample_size, processing_time, checkpoint_file, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_data['timestamp'],
                run_data['analysis_type'],
                run_data['scale'],
                run_data['primes_count'],
                run_data['gaps_count'],
                run_data['fft_sample_size'],
                run_data['autocorr_sample_size'],
                run_data['processing_time'],
                run_data['checkpoint_file'],
                run_data['status']
            ))

            run_id = cursor.lastrowid

            # Store harmonic ratios
            self._store_harmonic_ratios(cursor, run_id, results)

            # Store detection statistics
            self._store_detection_stats(cursor, run_id, results)

            # Store raw results
            self._store_raw_results(cursor, run_id, results)

            conn.commit()

        print(f"💾 Results stored in database: run_id={run_id}")
        return run_id

    def _store_harmonic_ratios(self, cursor, run_id, results):
        """Store individual harmonic ratio detections."""
        # FFT results
        if 'fft_analysis' in results and 'peaks' in results['fft_analysis']:
            for peak in results['fft_analysis']['peaks']:
                cursor.execute('''
                    INSERT INTO harmonic_ratios
                    (run_id, ratio_symbol, ratio_value, method, detected, distance, magnitude, peak_rank, frequency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    peak.get('closest_ratio', {}).get('symbol', 'unknown'),
                    peak.get('ratio', 0),
                    'fft',
                    peak.get('match', False),
                    peak.get('distance', 0),
                    peak.get('magnitude', 0),
                    peak.get('rank', 0),
                    peak.get('frequency', 0)
                ))

        # Autocorrelation results
        if 'autocorr_analysis' in results and 'peaks' in results['autocorr_analysis']:
            for peak in results['autocorr_analysis']['peaks']:
                cursor.execute('''
                    INSERT INTO harmonic_ratios
                    (run_id, ratio_symbol, ratio_value, method, detected, distance, correlation, peak_rank, lag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    peak.get('closest_ratio', {}).get('symbol', 'unknown'),
                    peak.get('ratio', 0),
                    'autocorr',
                    peak.get('match', False),
                    peak.get('distance', 0),
                    peak.get('correlation', 0),
                    peak.get('rank', 0),
                    peak.get('frequency', 0)
                ))

    def _store_detection_stats(self, cursor, run_id, results):
        """Store overall detection statistics."""
        if 'validation' in results:
            val = results['validation']
            cursor.execute('''
                INSERT INTO detection_stats
                (run_id, total_ratios, fft_detections, autocorr_detections,
                 consensus_detections, unique_ratios_detected, detection_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                8,  # Total known ratios
                val.get('fft_matches', 0),
                val.get('autocorr_matches', 0),
                val.get('common_ratios', 0),
                val.get('unique_ratios_detected', 0),
                (val.get('unique_ratios_detected', 0) / 8.0) * 100
            ))

    def _store_raw_results(self, cursor, run_id, results):
        """Store raw JSON results for detailed analysis."""
        cursor.execute('''
            INSERT INTO raw_results (run_id, result_type, result_data)
            VALUES (?, ?, ?)
        ''', (
            run_id,
            'full_results',
            json.dumps(results, default=str)
        ))

    def get_recent_runs(self, limit=10):
        """Get recent analysis runs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, analysis_type, scale, primes_count,
                       processing_time, status
                FROM analysis_runs
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            runs = []
            for row in cursor.fetchall():
                runs.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'analysis_type': row[2],
                    'scale': row[3],
                    'primes_count': row[4],
                    'processing_time': row[5],
                    'status': row[6]
                })

            return runs

    def get_run_details(self, run_id):
        """Get detailed results for a specific run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get run info
            cursor.execute('SELECT * FROM analysis_runs WHERE id = ?', (run_id,))
            run_row = cursor.fetchone()
            if not run_row:
                return None

            run_info = {
                'id': run_row[0],
                'timestamp': run_row[1],
                'analysis_type': run_row[2],
                'scale': run_row[3],
                'primes_count': run_row[4],
                'gaps_count': run_row[5],
                'fft_sample_size': run_row[6],
                'autocorr_sample_size': run_row[7],
                'processing_time': run_row[8],
                'checkpoint_file': run_row[9],
                'status': run_row[10]
            }

            # Get harmonic ratios
            cursor.execute('SELECT * FROM harmonic_ratios WHERE run_id = ?', (run_id,))
            ratios = []
            for row in cursor.fetchall():
                ratios.append({
                    'ratio_symbol': row[2],
                    'ratio_value': row[3],
                    'method': row[4],
                    'detected': bool(row[5]),
                    'distance': row[6],
                    'magnitude': row[7],
                    'correlation': row[8],
                    'peak_rank': row[9],
                    'frequency': row[10],
                    'lag': row[11]
                })

            # Get detection stats
            cursor.execute('SELECT * FROM detection_stats WHERE run_id = ?', (run_id,))
            stats_row = cursor.fetchone()
            if stats_row:
                stats = {
                    'total_ratios': stats_row[2],
                    'fft_detections': stats_row[3],
                    'autocorr_detections': stats_row[4],
                    'consensus_detections': stats_row[5],
                    'unique_ratios_detected': stats_row[6],
                    'detection_rate': stats_row[7]
                }
            else:
                stats = None

            return {
                'run_info': run_info,
                'harmonic_ratios': ratios,
                'detection_stats': stats
            }

    def get_detection_progression(self):
        """Get progression of harmonic detections across runs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get all runs with detection stats
            cursor.execute('''
                SELECT ar.timestamp, ar.scale, ds.unique_ratios_detected, ds.detection_rate
                FROM analysis_runs ar
                LEFT JOIN detection_stats ds ON ar.id = ds.run_id
                WHERE ds.unique_ratios_detected IS NOT NULL
                ORDER BY ar.timestamp
            ''')

            progression = []
            for row in cursor.fetchall():
                progression.append({
                    'timestamp': row[0],
                    'scale': row[1],
                    'unique_ratios': row[2],
                    'detection_rate': row[3]
                })

            return progression

    def display_database_summary(self):
        """Display a comprehensive summary of stored results."""
        print("🌌 WALLACE TRANSFORM RESULTS DATABASE")
        print("=" * 60)

        # Recent runs
        recent_runs = self.get_recent_runs(5)
        if recent_runs:
            print("📊 RECENT ANALYSIS RUNS:")
            print("ID | Timestamp | Type | Scale | Primes | Time | Status")
            print("-" * 65)
            for run in recent_runs:
                print("2d")

            print()

        # Detection progression
        progression = self.get_detection_progression()
        if progression:
            print("📈 HARMONIC DETECTION PROGRESSION:")
            print("Timestamp | Scale | Unique Ratios | Detection Rate")
            print("-" * 55)
            for entry in progression[-5:]:  # Last 5 entries
                print(f"{entry['timestamp'][:19]} | {entry['scale']:>5,} | {entry['unique_ratios']:>13} | {entry['detection_rate']:>13.1f}%")
            print()

        # Overall statistics
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total runs
            cursor.execute('SELECT COUNT(*) FROM analysis_runs')
            total_runs = cursor.fetchone()[0]

            # Total detections
            cursor.execute('SELECT COUNT(*) FROM harmonic_ratios WHERE detected = 1')
            total_detections = cursor.fetchone()[0]

            # Unique ratios found
            cursor.execute('SELECT COUNT(DISTINCT ratio_symbol) FROM harmonic_ratios WHERE detected = 1')
            unique_ratios = cursor.fetchone()[0]

            print("🏆 DATABASE STATISTICS:")
            print(f"   Total Analysis Runs: {total_runs}")
            print(f"   Total Harmonic Detections: {total_detections}")
            print(f"   Unique Ratios Discovered: {unique_ratios}")
            print(f"   Average Detections per Run: {total_detections/total_runs:.1f}")

    def export_results_csv(self, run_id, output_file=None):
        """Export results for a specific run to CSV."""
        if not output_file:
            output_file = f"wallace_run_{run_id}_export.csv"

        details = self.get_run_details(run_id)
        if not details:
            print(f"❌ Run {run_id} not found")
            return

        with open(output_file, 'w') as f:
            # Write header
            f.write("Ratio Symbol,Ratio Value,Method,Detected,Distance,Magnitude,Correlation,Peak Rank,Frequency,Lag\n")

            # Write data
            for ratio in details['harmonic_ratios']:
                f.write(f"{ratio['ratio_symbol']},{ratio['ratio_value']},{ratio['method']},{ratio['detected']},{ratio['distance']},{ratio['magnitude']},{ratio['correlation']},{ratio['peak_rank']},{ratio['frequency']},{ratio['lag']}\n")

        print(f"📊 Results exported to: {output_file}")

def main():
    """Database demonstration."""
    db = WallaceResultsDatabase()

    # Display current database status
    db.display_database_summary()

if __name__ == "__main__":
    main()
