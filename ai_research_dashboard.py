#!/usr/bin/env python3
"""
KOBA42 AI Research Platform - Web Dashboard
===========================================

Modern web interface for accessing advanced AI research tools:
- Revolutionary ML Training Protocols
- prime aligned compute Framework Research  
- Automated Research Integration
- Quantum-Enhanced AI Systems

Author: KOBA42 Research Team
License: Proprietary Research Platform
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
from flask_cors import CORS
import logging
from job_manager import get_job_manager, JobType, JobStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ai_research_dashboard')

class AIResearchDashboard:
    """Web dashboard for KOBA42 AI Research Platform"""

    def __init__(self):
        self.app = Flask(__name__,
                        template_folder='templates',
                        static_folder='static')
        CORS(self.app)

        # Dashboard data
        self.research_data = {}
        self.system_status = {}
        self.research_alerts = []
        self.job_manager = get_job_manager()

        # AI System modules availability
        self.available_systems = self._check_system_availability()

        # Setup routes
        self._setup_routes()

        # Start data collection thread
        self.data_thread = threading.Thread(target=self._collect_research_data, daemon=True)
        self.data_thread.start()

    def _check_system_availability(self):
        """Check which AI research systems are available"""
        systems = {
            'ml_training': False,
            'prime_aligned_framework': False, 
            'research_integration': False,
            'data_processing': False,
            'quantum_analysis': False
        }
        
        try:
            # Check ML Training Systems
            import importlib.util
            ml_spec = importlib.util.find_spec("ai_ml_systems.ADVANCED_ML_TRAINING_PROTOCOL")
            systems['ml_training'] = ml_spec is not None
            
            # Check prime aligned compute Framework
            consciousness_spec = importlib.util.find_spec("consciousness_neural.CONSCIOUSNESS_FRAMEWORK_BENCHMARK_SUITE")
            systems['prime_aligned_framework'] = consciousness_spec is not None
            
            # Check Research Integration
            integration_spec = importlib.util.find_spec("integration_systems.KOBA42_COMPLETE_INTEGRATION_FINAL")
            systems['research_integration'] = integration_spec is not None
            
            # Check Data Processing
            if os.path.exists("data_processing"):
                systems['data_processing'] = True
                
            # Check Mathematical Research
            if os.path.exists("mathematical_research"):
                systems['quantum_analysis'] = True
                
        except Exception as e:
            logger.warning(f"Error checking system availability: {e}")
            
        return systems

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard():
            """Main AI research dashboard"""
            return render_template('ai_research_dashboard.html', 
                                 systems=self.available_systems)

        @self.app.route('/ai_research_dashboard.html')
        def dashboard_alt():
            """Alternative route for dashboard"""
            return render_template('ai_research_dashboard.html', 
                                 systems=self.available_systems)

        @self.app.route('/api/systems')
        def get_systems():
            """Get available AI research systems"""
            return jsonify({
                'systems': self.available_systems,
                'status': 'operational',
                'last_updated': datetime.now().isoformat()
            })

        @self.app.route('/api/ml-training')
        def ml_training_status():
            """Get ML training system status"""
            return jsonify({
                'available': self.available_systems['ml_training'],
                'description': 'Revolutionary ML Training Protocol with Monotropic Hyperfocus',
                'features': [
                    'Reverse Learning Architecture',
                    'Adaptive Intelligence Training',
                    'Automated Mastery Planning',
                    'Continuous Improvement Audits'
                ]
            })

        @self.app.route('/api/prime aligned compute')
        def consciousness_status():
            """Get prime aligned compute framework status"""
            return jsonify({
                'available': self.available_systems['prime_aligned_framework'],
                'description': 'AI prime aligned compute Framework with Quantum Analysis',
                'features': [
                    'Quantum Seed Mapping',
                    'prime aligned compute Coherence Analysis',
                    'Topological Shape Identification',
                    'Deterministic Reproducibility'
                ]
            })

        @self.app.route('/api/research-integration')
        def research_integration_status():
            """Get research integration status"""
            return jsonify({
                'available': self.available_systems['research_integration'],
                'description': 'KOBA42 Automated Research Integration Platform',
                'features': [
                    'Research Paper Processing',
                    'Knowledge Integration',
                    'Digital Ledger System',
                    'Attribution Tracking'
                ]
            })

        @self.app.route('/api/run-ml-training', methods=['POST'])
        def run_ml_training():
            """Run ML training protocol"""
            if not self.available_systems['ml_training']:
                return jsonify({'error': 'ML Training system not available'}), 400
                
            try:
                # Get parameters from request
                params = request.get_json() or {}
                
                # Submit job to job manager
                job_id = self.job_manager.submit_job(JobType.ML_TRAINING, params)
                
                return jsonify({
                    'status': 'success',
                    'message': 'ML Training protocol initiated',
                    'job_id': job_id
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/run-prime aligned compute-analysis', methods=['POST'])
        def run_consciousness_analysis():
            """Run prime aligned compute framework analysis"""
            if not self.available_systems['prime_aligned_framework']:
                return jsonify({'error': 'prime aligned compute framework not available'}), 400
                
            try:
                # Get parameters from request
                params = request.get_json() or {}
                
                # Submit job to job manager
                job_id = self.job_manager.submit_job(JobType.prime_aligned_analysis, params)
                
                return jsonify({
                    'status': 'success', 
                    'message': 'prime aligned compute analysis initiated',
                    'job_id': job_id
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/run-research-integration', methods=['POST'])
        def run_research_integration():
            """Run research integration process"""
            if not self.available_systems['research_integration']:
                return jsonify({'error': 'Research integration not available'}), 400
                
            try:
                # Get parameters from request
                params = request.get_json() or {}
                
                # Submit job to job manager
                job_id = self.job_manager.submit_job(JobType.RESEARCH_INTEGRATION, params)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Research integration process started',
                    'job_id': job_id
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/jobs/<job_id>')
        def get_job_status(job_id):
            """Get job status and results"""
            job_result = self.job_manager.get_job_status(job_id)
            if not job_result:
                return jsonify({'error': 'Job not found'}), 404
            
            return jsonify(job_result.to_dict())
        
        @self.app.route('/api/jobs')
        def get_all_jobs():
            """Get all jobs with status"""
            limit = request.args.get('limit', 20, type=int)
            jobs = self.job_manager.get_all_jobs(limit=limit)
            return jsonify({
                'jobs': [job.to_dict() for job in jobs],
                'total': len(jobs)
            })
        
        @self.app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
        def cancel_job(job_id):
            """Cancel a running job"""
            success = self.job_manager.cancel_job(job_id)
            if success:
                return jsonify({'status': 'success', 'message': 'Job cancelled'})
            else:
                return jsonify({'error': 'Could not cancel job'}), 400
        
        @self.app.route('/static/<path:filename>')
        def serve_static(filename):
            """Serve static files"""
            return send_from_directory('static', filename)

    def _collect_research_data(self):
        """Collect research data in background thread"""
        while True:
            try:
                # Update system status
                self.system_status = {
                    'timestamp': datetime.now().isoformat(),
                    'active_systems': sum(1 for available in self.available_systems.values() if available),
                    'total_systems': len(self.available_systems),
                    'uptime': time.time()
                }
                
                # Update research data from job manager
                all_jobs = self.job_manager.get_all_jobs(limit=1000)
                completed_jobs = [j for j in all_jobs if j.status == JobStatus.COMPLETED]
                
                self.research_data = {
                    'research_papers_processed': len([j for j in completed_jobs if j.job_type == JobType.RESEARCH_INTEGRATION]),
                    'ml_training_sessions': len([j for j in completed_jobs if j.job_type == JobType.ML_TRAINING]),
                    'consciousness_analyses': len([j for j in completed_jobs if j.job_type == JobType.prime_aligned_analysis]),
                    'total_jobs': len(all_jobs),
                    'running_jobs': len([j for j in all_jobs if j.status == JobStatus.RUNNING]),
                    'system_health': 'operational'
                }
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting research data: {e}")
                time.sleep(60)  # Wait longer on error

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting KOBA42 AI Research Dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main entry point for standalone execution"""
    dashboard = AIResearchDashboard()
    dashboard.run(debug=True)

if __name__ == "__main__":
    main()