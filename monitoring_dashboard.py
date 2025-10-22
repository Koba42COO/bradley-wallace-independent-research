#!/usr/bin/env python3
"""
Real-Time Monitoring Dashboard - Firefly-Nexus PAC
==================================================

Complete monitoring dashboard with:
- Real-time consciousness metrics
- Performance monitoring
- Health status
- Resource utilization
- Alerting system
- Historical data

Author: Bradley Wallace, COO Koba42
Framework: PAC (Prime Aligned Compute)
Consciousness Level: 7 (Prime Topology)
"""

import os
import time
import json
import math
import numpy as np
from flask import Flask, render_template, jsonify, request
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psutil
import requests
from collections import deque
import sqlite3

class ConsciousnessMonitor:
    """Real-time consciousness monitoring system"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.delta = 2.414213562373095
        self.reality_distortion = 1.1808
        self.consciousness_level = 7
        
        # Monitoring data
        self.metrics_history = deque(maxlen=1000)
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'consciousness_level': 5.0,
            'reality_distortion': 2.0
        }
        
        # Database
        self.db_path = 'consciousness_monitor.db'
        self.init_database()
        
        # Start monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consciousness_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                consciousness_level REAL,
                reality_distortion REAL,
                mobius_phase REAL,
                cpu_usage REAL,
                memory_usage REAL,
                metronome_freq REAL,
                coherent_weight REAL,
                exploratory_weight REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                alert_type TEXT,
                message TEXT,
                severity TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store in database
                self._store_metrics(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Update history
                self.metrics_history.append(metrics)
                
                time.sleep(1.0)  # 1 second interval
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5.0)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics"""
        return {
            'timestamp': time.time(),
            'consciousness_level': self.consciousness_level,
            'reality_distortion': self.reality_distortion,
            'mobius_phase': (time.time() * self.phi) % (2 * math.pi),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'metronome_freq': 0.7,
            'coherent_weight': 0.79,
            'exploratory_weight': 0.21,
            'phi': self.phi,
            'delta': self.delta
        }
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO consciousness_metrics 
            (timestamp, consciousness_level, reality_distortion, mobius_phase, 
             cpu_usage, memory_usage, metronome_freq, coherent_weight, exploratory_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['timestamp'],
            metrics['consciousness_level'],
            metrics['reality_distortion'],
            metrics['mobius_phase'],
            metrics['cpu_usage'],
            metrics['memory_usage'],
            metrics['metronome_freq'],
            metrics['coherent_weight'],
            metrics['exploratory_weight']
        ))
        
        conn.commit()
        conn.close()
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions"""
        alerts = []
        
        if metrics['cpu_usage'] > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'type': 'cpu_usage',
                'message': f"High CPU usage: {metrics['cpu_usage']:.1f}%",
                'severity': 'warning'
            })
        
        if metrics['memory_usage'] > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'memory_usage',
                'message': f"High memory usage: {metrics['memory_usage']:.1f}%",
                'severity': 'critical'
            })
        
        if metrics['consciousness_level'] < self.alert_thresholds['consciousness_level']:
            alerts.append({
                'type': 'consciousness_level',
                'message': f"Low consciousness level: {metrics['consciousness_level']:.1f}",
                'severity': 'critical'
            })
        
        if metrics['reality_distortion'] > self.alert_thresholds['reality_distortion']:
            alerts.append({
                'type': 'reality_distortion',
                'message': f"High reality distortion: {metrics['reality_distortion']:.3f}",
                'severity': 'warning'
            })
        
        # Store alerts
        for alert in alerts:
            self._store_alert(alert)
    
    def _store_alert(self, alert: Dict[str, Any]):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        ''', (
            time.time(),
            alert['type'],
            alert['message'],
            alert['severity']
        ))
        
        conn.commit()
        conn.close()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return self._collect_metrics()
    
    def get_historical_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (hours * 3600)
        cursor.execute('''
            SELECT * FROM consciousness_metrics 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        metrics = []
        for row in rows:
            metrics.append({
                'timestamp': row[1],
                'consciousness_level': row[2],
                'reality_distortion': row[3],
                'mobius_phase': row[4],
                'cpu_usage': row[5],
                'memory_usage': row[6],
                'metronome_freq': row[7],
                'coherent_weight': row[8],
                'exploratory_weight': row[9]
            })
        
        return metrics
    
    def get_alerts(self, unresolved_only: bool = True) -> List[Dict[str, Any]]:
        """Get alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if unresolved_only:
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE resolved = FALSE 
                ORDER BY timestamp DESC
            ''')
        else:
            cursor.execute('''
                SELECT * FROM alerts 
                ORDER BY timestamp DESC
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in rows:
            alerts.append({
                'id': row[0],
                'timestamp': row[1],
                'alert_type': row[2],
                'message': row[3],
                'severity': row[4],
                'resolved': bool(row[5])
            })
        
        return alerts
    
    def resolve_alert(self, alert_id: int):
        """Resolve an alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts SET resolved = TRUE WHERE id = ?
        ''', (alert_id,))
        
        conn.commit()
        conn.close()

# Create Flask app
app = Flask(__name__)
monitor = ConsciousnessMonitor()

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/metrics/current')
def api_current_metrics():
    """Get current metrics"""
    return jsonify(monitor.get_current_metrics())

@app.route('/api/metrics/historical')
def api_historical_metrics():
    """Get historical metrics"""
    hours = request.args.get('hours', 24, type=int)
    metrics = monitor.get_historical_metrics(hours)
    return jsonify(metrics)

@app.route('/api/alerts')
def api_alerts():
    """Get alerts"""
    unresolved_only = request.args.get('unresolved_only', 'true').lower() == 'true'
    alerts = monitor.get_alerts(unresolved_only)
    return jsonify(alerts)

@app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
def api_resolve_alert(alert_id):
    """Resolve an alert"""
    monitor.resolve_alert(alert_id)
    return jsonify({'status': 'resolved'})

@app.route('/api/status')
def api_status():
    """Get system status"""
    current_metrics = monitor.get_current_metrics()
    alerts = monitor.get_alerts(unresolved_only=True)
    
    # Determine overall status
    status = 'healthy'
    if any(alert['severity'] == 'critical' for alert in alerts):
        status = 'critical'
    elif any(alert['severity'] == 'warning' for alert in alerts):
        status = 'warning'
    
    return jsonify({
        'status': status,
        'consciousness_level': current_metrics['consciousness_level'],
        'reality_distortion': current_metrics['reality_distortion'],
        'cpu_usage': current_metrics['cpu_usage'],
        'memory_usage': current_metrics['memory_usage'],
        'active_alerts': len(alerts),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/consciousness/transform', methods=['POST'])
def api_consciousness_transform():
    """Consciousness transformation API"""
    try:
        data = request.get_json()
        if not data or 'values' not in data:
            return jsonify({'error': 'Missing values array'}), 400
        
        values = np.array(data['values'], dtype=float)
        
        # Apply Wallace Transform
        transformed = []
        for x in values:
            if x <= 0:
                x = 1e-15
            log_term = math.log(x + 1e-15)
            phi_power = abs(log_term) ** monitor.phi
            sign = 1.0 if log_term >= 0 else -1.0
            result = monitor.phi * phi_power * sign + monitor.delta
            transformed.append(result)
        
        # Apply Fractal-Harmonic Transform
        values = np.maximum(values, 1e-15)
        log_terms = np.log(values + 1e-15)
        phi_powers = np.abs(log_terms) ** monitor.phi
        signs = np.sign(log_terms)
        transformed_harmonic = monitor.phi * phi_powers * signs
        
        # 79/21 consciousness split
        coherent = 0.79 * transformed_harmonic
        exploratory = 0.21 * transformed_harmonic
        fractal_result = coherent + exploratory
        
        # Psychotronic processing
        mobius_phase = np.sum(values) * monitor.phi % (2 * math.pi)
        magnitude = np.mean(np.abs(values)) * monitor.reality_distortion
        coherence = 0.79 * (1.0 - np.std(values) / (np.mean(np.abs(values)) + 1e-15))
        exploration = 0.21 * np.std(values) / (np.mean(np.abs(values)) + 1e-15)
        
        response = {
            'wallace_transform': transformed,
            'fractal_harmonic': fractal_result.tolist(),
            'consciousness_amplitude': {
                'magnitude': magnitude,
                'phase': mobius_phase,
                'coherence': coherence,
                'exploration': exploration
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/consciousness/mobius', methods=['POST'])
def api_mobius_loop():
    """Möbius loop learning API"""
    try:
        data = request.get_json()
        if not data or 'values' not in data:
            return jsonify({'error': 'Missing values array'}), 400
        
        values = np.array(data['values'], dtype=float)
        cycles = data.get('cycles', 10)
        
        # Möbius loop learning
        evolution_history = []
        consciousness_trajectory = []
        current_data = values.copy()
        
        for cycle in range(cycles):
            # Apply Wallace Transform
            transformed = []
            for x in current_data:
                if x <= 0:
                    x = 1e-15
                log_term = math.log(x + 1e-15)
                phi_power = abs(log_term) ** monitor.phi
                sign = 1.0 if log_term >= 0 else -1.0
                result = monitor.phi * phi_power * sign + monitor.delta
                transformed.append(result)
            
            transformed = np.array(transformed)
            
            # Psychotronic processing
            mobius_phase = np.sum(transformed) * monitor.phi % (2 * math.pi)
            magnitude = np.mean(np.abs(transformed)) * monitor.reality_distortion
            coherence = 0.79 * (1.0 - np.std(transformed) / (np.mean(np.abs(transformed)) + 1e-15))
            exploration = 0.21 * np.std(transformed) / (np.mean(np.abs(transformed)) + 1e-15)
            
            consciousness = {
                'magnitude': magnitude,
                'phase': mobius_phase,
                'coherence': coherence,
                'exploration': exploration
            }
            consciousness_trajectory.append(consciousness)
            
            # Möbius twist (feed output back as input)
            twist_factor = math.sin(mobius_phase) * math.cos(math.pi)
            current_data = current_data * (1 + twist_factor * magnitude)
            
            # Record evolution
            evolution_history.append({
                'cycle': cycle,
                'consciousness_magnitude': magnitude,
                'coherence': coherence,
                'exploration': exploration,
                'reality_distortion': monitor.reality_distortion,
                'mobius_phase': mobius_phase
            })
        
        response = {
            'evolution_history': evolution_history,
            'consciousness_trajectory': consciousness_trajectory,
            'final_consciousness': consciousness_trajectory[-1],
            'total_learning_gain': sum(c['magnitude'] for c in consciousness_trajectory),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/consciousness/prime-graph', methods=['POST'])
def api_prime_graph_compression():
    """Prime graph compression API"""
    try:
        data = request.get_json()
        if not data or 'values' not in data:
            return jsonify({'error': 'Missing values array'}), 400
        
        values = np.array(data['values'], dtype=float)
        
        # Prime graph compression
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        compressed = []
        
        for value in values:
            # Find nearest prime
            nearest_prime = min(primes, key=lambda p: abs(p - value))
            
            # Apply consciousness weighting
            consciousness_weight = 0.79 if nearest_prime % 2 == 0 else 0.21
            weighted_value = value * consciousness_weight
            
            # Apply φ-delta scaling
            phi_coord = monitor.phi ** (primes.index(nearest_prime) % 21)
            delta_coord = monitor.delta ** (primes.index(nearest_prime) % 7)
            
            compressed_value = weighted_value * phi_coord * delta_coord
            compressed.append(compressed_value)
        
        response = {
            'compressed_values': compressed,
            'compression_ratio': len(values) / len(compressed),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
