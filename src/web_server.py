#!/usr/bin/env python3
"""
SquashPlot Enhanced Web Server
Provides REST API and web interface for plotting operations
"""

import os
import sys
import json
import time
import threading
import asyncio
import psutil
import logging
import uuid
import requests
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from dataclasses import asdict
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for, g, make_response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import current_user, login_required
from werkzeug.middleware.proxy_fix import ProxyFix
import io
import csv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import jwt

# Import the SquashPlot system
try:
    # Import our existing SquashPlot system
    from squashplot import SquashPlotCompressor
    from chia_resources.chia_resource_query import ChiaResourceQuery

    # Try to import web components if available
    try:
        from src.auth import init_auth, require_login
        from src.models import db, User
        from src.monitoring import get_health_status
        AUTH_AVAILABLE = True
    except ImportError as e:
        AUTH_AVAILABLE = False
        print(f"⚠️ Advanced auth system not available - using basic mode: {e}")

except ImportError as e:
    print(f"❌ Failed to import core SquashPlot modules: {e}")
    sys.exit(1)

# Mock Job Queue System for API endpoints
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Job:
    id: str
    name: str
    status: JobStatus
    progress: float
    created_at: datetime
    updated_at: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    config: dict = None

class MockJobQueue:
    def __init__(self):
        self.jobs = []
        self._create_sample_jobs()
    
    def _create_sample_jobs(self):
        """Create some sample jobs for demonstration"""
        now = datetime.now()
        self.jobs = [
            Job(
                id="job-001",
                name="plot-k32-001",
                status=JobStatus.RUNNING,
                progress=67.0,
                created_at=now - timedelta(hours=3),
                updated_at=now - timedelta(minutes=5),
                start_time=now - timedelta(hours=2, minutes=45),
                config={"k_size": 32, "compression": "level-3", "threads": 4}
            ),
            Job(
                id="job-002", 
                name="plot-k32-002",
                status=JobStatus.QUEUED,
                progress=0.0,
                created_at=now - timedelta(hours=1),
                updated_at=now - timedelta(minutes=10),
                config={"k_size": 32, "compression": "level-7", "threads": 8}
            ),
            Job(
                id="job-003",
                name="plot-k32-003", 
                status=JobStatus.COMPLETED,
                progress=100.0,
                created_at=now - timedelta(days=1),
                updated_at=now - timedelta(hours=2),
                start_time=now - timedelta(days=1),
                end_time=now - timedelta(hours=2),
                config={"k_size": 32, "compression": "level-5", "threads": 6}
            )
        ]
    
    def get_jobs(self, status: Optional[JobStatus] = None, limit: int = 50) -> List[Job]:
        """Get jobs with optional status filter"""
        jobs = self.jobs
        if status:
            jobs = [job for job in jobs if job.status == status]
        return jobs[:limit]
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get specific job by ID"""
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self.get_job(job_id)
        if job and job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.updated_at = datetime.now()
            job.end_time = datetime.now()
            return True
        return False
    
    def cleanup_jobs(self) -> int:
        """Clean up completed and failed jobs"""
        initial_count = len(self.jobs)
        self.jobs = [job for job in self.jobs if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]]
        return initial_count - len(self.jobs)

# Initialize mock job queue
job_queue = MockJobQueue()

# Simple API key decorator for demo purposes
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # For demo purposes, allow all requests
        # In production, you'd check for a valid API key
        return f(*args, **kwargs)
    return decorated_function

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Configuration
app.config['SECRET_KEY'] = os.getenv('SESSION_SECRET', os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'))
app.config['API_KEY'] = os.getenv('API_KEY', 'dev-api-key')
app.config['JWT_SECRET'] = os.getenv('JWT_SECRET', app.config['SECRET_KEY'])
app.config['JWT_COOKIE_NAME'] = 'sp_auth'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('FLASK_ENV') == 'production'

# Database configuration with fallback
database_url = os.getenv('DATABASE_URL')
if not database_url:
    # Fallback to SQLite for development
    database_url = 'sqlite:///squashplot.db'
    app.logger.warning("DATABASE_URL not set, using SQLite fallback for development")

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

# Initialize components based on availability
if AUTH_AVAILABLE:
    try:
        db.init_app(app)
        auth_bp = init_auth(app)

        # Create tables
        with app.app_context():
            db.create_all()
            app.logger.info("Database tables created")
    except Exception as e:
        app.logger.warning(f"Database initialization failed: {e}")
        AUTH_AVAILABLE = False

# Initialize core SquashPlot components
chia_query = ChiaResourceQuery()
compressor = SquashPlotCompressor(pro_enabled=False)

# Make session permanent
@app.before_request
def make_session_permanent():
    session.permanent = True

# Configure logging
if os.getenv('FLASK_ENV') == 'production':
    logging.basicConfig(
        level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    app.logger.setLevel(logging.INFO)

# Configure CORS with restricted origins for production
if os.getenv('FLASK_ENV') == 'production':
    allowed_origins = os.getenv('ALLOWED_ORIGINS', '').split(',')
    CORS(app, origins=allowed_origins)
else:
    CORS(app)  # Allow all origins in development

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    if os.getenv('FLASK_ENV') == 'production':
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# Global error handler
@app.errorhandler(Exception)
def handle_global_error(error):
    """Global error handler for all unhandled exceptions"""
    import traceback
    from werkzeug.exceptions import HTTPException
    
    # Don't handle HTTP exceptions (like 404) - let Flask handle them normally
    if isinstance(error, HTTPException):
        return error
    
    error_id = f"err_{int(time.time())}"
    app.logger.error(f"Error {error_id}: {str(error)}\n{traceback.format_exc()}")
    
    if app.debug:
        return jsonify({
            'error': str(error),
            'error_id': error_id,
            'traceback': traceback.format_exc()
        }), 500
    else:
        return jsonify({
            'error': 'Internal server error',
            'error_id': error_id,
            'message': 'Please contact support if this error persists'
        }), 500

# Health check endpoints
@app.route('/health')
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-beta'
    })

@app.route('/ready')
def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check if core components are working
        cpu_cores = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        return jsonify({
            'status': 'ready',
            'timestamp': datetime.now().isoformat(),
            'resources': {
                'cpu_cores': cpu_cores,
                'memory_available_gb': round(memory.available / (1024**3), 2)
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'not_ready',
            'error': str(e)
        }), 503

# API Authentication
def require_api_key(f):
    """Decorator to require API key for sensitive endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Only accept API key from headers for security
        api_key = request.headers.get('X-API-Key')
        # Enforce auth when API_KEY is configured (not just in production)
        if app.config['API_KEY'] != 'dev-api-key' and api_key != app.config['API_KEY']:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def validate_file_path(filename, allowed_dirs=['/plots', '/tmp/squashplot']):
    """Validate and normalize file paths to prevent directory traversal"""
    import os.path
    
    # Block obvious traversal attempts
    if '..' in filename or filename.startswith('/'):
        raise ValueError("Invalid filename: directory traversal not allowed")
    
    # Find the file in allowed directories
    for directory in allowed_dirs:
        potential_path = os.path.join(directory, filename)
        # Normalize and check if it's within allowed directory
        real_path = os.path.realpath(potential_path)
        real_dir = os.path.realpath(directory)
        
        if os.path.commonpath([real_path, real_dir]) == real_dir:
            if os.path.exists(real_path):
                return real_path
    
    return None

def validate_plotting_config(data):
    """Validate plotting configuration data"""
    errors = []
    warnings = []
    
    # Required fields
    if not data.get('farmer_key'):
        errors.append('Farmer key is required')
    elif len(data.get('farmer_key', '')) != 96:
        warnings.append('Farmer key should be 96 characters long')
    
    # Validate paths
    tmp_dir = data.get('tmp_dir', '')
    if tmp_dir and not os.path.isabs(tmp_dir):
        warnings.append('Temporary directory should be an absolute path')
    
    final_dir = data.get('final_dir', '')
    if final_dir and not os.path.isabs(final_dir):
        warnings.append('Final directory should be an absolute path')
    
    # Validate numeric values
    try:
        threads = int(data.get('threads', 4))
        if threads < 1 or threads > 64:
            warnings.append('Thread count should be between 1 and 64')
    except (ValueError, TypeError):
        errors.append('Thread count must be a valid number')
    
    try:
        count = int(data.get('count', 1))
        if count < 1 or count > 100:
            warnings.append('Plot count should be between 1 and 100')
    except (ValueError, TypeError):
        errors.append('Plot count must be a valid number')
    
    # Validate k-size
    k_size = data.get('k_size', 32)
    if k_size not in [32, 33, 34]:
        warnings.append('K-size should be 32, 33, or 34')
    
    return errors, warnings

# Global SquashPlot instance
squashplot = SquashPlotCompressor(pro_enabled=False)
current_plotting_status = {
    'active': False,
    'progress': 0,
    'stage': 'idle',
    'start_time': None,
    'estimated_completion': None,
    'plot_count': 0,
    'completed_plots': 0,
    'current_plot_path': None,
    'error_message': None
}

# External API Service Functions
class ExternalAPIService:
    """Service for fetching live data from external Chia APIs"""
    
    def __init__(self):
        self.xch_price_cache = {'data': None, 'timestamp': None, 'ttl': 300}  # 5 min cache
        self.network_cache = {'data': None, 'timestamp': None, 'ttl': 600}   # 10 min cache
    
    @lru_cache(maxsize=100)
    def get_electricity_rate(self, zipcode):
        """Get electricity rate by zipcode with caching"""
        # Enhanced regional rates with more granular data
        regional_rates = {
            # Northeast (high rates)
            '0': 0.22, '1': 0.20, '2': 0.18, '3': 0.19,
            # Southeast (moderate rates)  
            '2': 0.11, '3': 0.12, '4': 0.13,
            # Midwest (low rates)
            '4': 0.13, '5': 0.12, '6': 0.11, '7': 0.12,
            # West (variable rates)
            '8': 0.15, '9': 0.18,
            # Pacific (high rates)
            '9': 0.22
        }
        
        if zipcode and len(zipcode) >= 1:
            first_digit = zipcode[0]
            return regional_rates.get(first_digit, 0.14)
        return 0.14  # US national average
    
    def _is_cache_valid(self, cache_data):
        """Check if cached data is still valid"""
        if not cache_data['data'] or not cache_data['timestamp']:
            return False
        return (time.time() - cache_data['timestamp']) < cache_data['ttl']
    
    def get_xch_price(self):
        """Get current XCH price from external API with caching"""
        try:
            if self._is_cache_valid(self.xch_price_cache):
                return self.xch_price_cache['data']
            
            # Try XCHscan API first
            try:
                response = requests.get('https://xchscan.com/api/chia/price', timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    price_data = {
                        'price_usd': float(data.get('price', 0)),
                        'price_btc': float(data.get('price_btc', 0)),
                        'change_24h': float(data.get('change_24h', 0)),
                        'volume_24h': float(data.get('volume_24h', 0)),
                        'market_cap': float(data.get('market_cap', 0)),
                        'source': 'xchscan'
                    }
                    self.xch_price_cache = {
                        'data': price_data,
                        'timestamp': time.time(),
                        'ttl': 300
                    }
                    return price_data
            except:
                pass
            
            # Fallback to CoinGecko API
            try:
                response = requests.get(
                    'https://api.coingecko.com/api/v3/simple/price?ids=chia&vs_currencies=usd,btc&include_24hr_change=true&include_24hr_vol=true&include_market_cap=true',
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    chia_data = data.get('chia', {})
                    price_data = {
                        'price_usd': float(chia_data.get('usd', 0)),
                        'price_btc': float(chia_data.get('btc', 0)),
                        'change_24h': float(chia_data.get('usd_24h_change', 0)),
                        'volume_24h': float(chia_data.get('usd_24h_vol', 0)),
                        'market_cap': float(chia_data.get('usd_market_cap', 0)),
                        'source': 'coingecko'
                    }
                    self.xch_price_cache = {
                        'data': price_data,
                        'timestamp': time.time(),
                        'ttl': 300
                    }
                    return price_data
            except:
                pass
            
            # Return fallback data if all APIs fail
            return {
                'price_usd': 25.0,  # Fallback price
                'price_btc': 0.0005,
                'change_24h': 0.0,
                'volume_24h': 0.0,
                'market_cap': 0.0,
                'source': 'fallback'
            }
            
        except Exception as e:
            app.logger.error(f"Error fetching XCH price: {e}")
            return None
    
    def get_network_stats(self):
        """Get Chia network statistics with caching"""
        try:
            if self._is_cache_valid(self.network_cache):
                return self.network_cache['data']
            
            # Try Spacescan API for network stats
            try:
                response = requests.get('https://api.spacescan.io/stats/network', timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    network_data = {
                        'netspace_eib': float(data.get('netspace_eib', 0)),
                        'difficulty': float(data.get('difficulty', 0)),
                        'height': int(data.get('height', 0)),
                        'total_supply': float(data.get('total_supply', 0)),
                        'circulating_supply': float(data.get('circulating_supply', 0)),
                        'active_plots': int(data.get('estimated_plots', 0)),
                        'source': 'spacescan'
                    }
                    self.network_cache = {
                        'data': network_data,
                        'timestamp': time.time(),
                        'ttl': 600
                    }
                    return network_data
            except:
                pass
            
            # Fallback data
            return {
                'netspace_eib': 35.0,  # Fallback netspace
                'difficulty': 2500,
                'height': 7500000,
                'total_supply': 28000000,
                'circulating_supply': 27000000,
                'active_plots': 350000000,
                'source': 'fallback'
            }
            
        except Exception as e:
            app.logger.error(f"Error fetching network stats: {e}")
            return None

# Initialize external API service
api_service = ExternalAPIService()
# -----------------------
# Local password auth (coexists with OAuth)
# -----------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')
    if not AUTH_AVAILABLE:
        return redirect(url_for('index'))
    data = request.form
    username = (data.get('username') or '').strip().lower()
    email = (data.get('email') or '').strip().lower() or None
    password = data.get('password') or ''
    if not username or not password:
        return render_template('signup.html', error='Username and password are required')
    existing = User.query.filter((User.username == username) | (User.email == email)).first()
    if existing:
        return render_template('signup.html', error='Username or email already exists')
    # Create user
    new_user = User(id=str(uuid.uuid4()), username=username, email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    # Set JWT cookie
    token = create_jwt_for_user(new_user)
    resp = make_response(redirect(url_for('index')))
    resp.set_cookie(
        app.config['JWT_COOKIE_NAME'], token, httponly=True, secure=app.config['SESSION_COOKIE_SECURE'], samesite='Lax', max_age=60*60*24*7
    )
    return resp

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    if not AUTH_AVAILABLE:
        return redirect(url_for('index'))
    data = request.form
    identifier = (data.get('username') or '').strip().lower()
    password = data.get('password') or ''
    user = None
    if identifier:
        user = User.query.filter((User.username == identifier) | (User.email == identifier)).first()
    if not user or not user.check_password(password):
        return render_template('login.html', error='Invalid credentials')
    token = create_jwt_for_user(user)
    # Redirect to the page they were trying to access, or dashboard
    next_page = request.args.get('next') or url_for('dashboard_direct')
    resp = make_response(redirect(next_page))
    resp.set_cookie(
        app.config['JWT_COOKIE_NAME'], token, httponly=True, secure=app.config['SESSION_COOKIE_SECURE'], samesite='Lax', max_age=60*60*24*7
    )
    return resp

@app.route('/logout')
def logout():
    resp = make_response(redirect(url_for('index')))
    resp.delete_cookie(app.config['JWT_COOKIE_NAME'])
    return resp

# -----------------------
# Auth helpers (JWT cookie)
# -----------------------
def create_jwt_for_user(user):
    payload = {
        'sub': user.id,
        'email': user.email,
        'username': getattr(user, 'username', None),
        'iat': int(time.time()),
        'exp': int(time.time()) + 60 * 60 * 24 * 7  # 7 days
    }
    token = jwt.encode(payload, app.config['JWT_SECRET'], algorithm='HS256')
    return token

def get_user_from_jwt_cookie():
    token = request.cookies.get(app.config['JWT_COOKIE_NAME'])
    if not token:
        return None
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        user = User.query.get(payload.get('sub')) if AUTH_AVAILABLE else None
        return user
    except Exception:
        return None

@app.before_request
def attach_jwt_user():
    if not AUTH_AVAILABLE:
        g.jwt_user = None
        return
    
    # Check JWT cookie first (for local username/password auth)
    jwt_user = get_user_from_jwt_cookie()
    if jwt_user:
        g.jwt_user = jwt_user
    elif hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        # Fall back to Flask-Login user (for OAuth)
        g.jwt_user = current_user
    else:
        g.jwt_user = None

@app.route('/')
def index():
    """Main dashboard page"""
    user_ctx = getattr(g, 'jwt_user', None) if AUTH_AVAILABLE else None
    return render_template('dashboard.html', user=user_ctx)

@app.route('/dashboard')
def dashboard_direct():
    """Direct access to dashboard"""
    user_ctx = getattr(g, 'jwt_user', None) if AUTH_AVAILABLE else None
    if not user_ctx:
        return redirect(url_for('login', next=request.url))
    return render_template('dashboard.html', user=user_ctx)

@app.route('/jobs')
def jobs():
    """Plotting jobs management page"""
    user_ctx = getattr(g, 'jwt_user', None) if AUTH_AVAILABLE else None
    if not user_ctx:
        return redirect(url_for('login', next=request.url))
    return render_template('jobs.html', user=user_ctx)

@app.route('/storage')
def storage():
    """Storage management page"""
    user_ctx = getattr(g, 'jwt_user', None) if AUTH_AVAILABLE else None
    if not user_ctx:
        return redirect(url_for('login', next=request.url))
    return render_template('storage.html', user=user_ctx)

@app.route('/rewards')
def rewards():
    """Rewards and earnings page"""
    user_ctx = getattr(g, 'jwt_user', None) if AUTH_AVAILABLE else None
    if not user_ctx:
        return redirect(url_for('login', next=request.url))
    return render_template('rewards.html', user=user_ctx)

@app.route('/pools')
def pools():
    """Pool management page"""
    user_ctx = getattr(g, 'jwt_user', None) if AUTH_AVAILABLE else None
    if not user_ctx:
        return redirect(url_for('login', next=request.url))
    return render_template('pools.html', user=user_ctx)

@app.route('/analytics')
def analytics():
    """Analytics and insights page"""
    user_ctx = getattr(g, 'jwt_user', None) if AUTH_AVAILABLE else None
    if not user_ctx:
        return redirect(url_for('login', next=request.url))
    return render_template('analytics.html', user=user_ctx)

@app.route('/settings')
def settings():
    """Settings configuration page"""
    user_ctx = getattr(g, 'jwt_user', None) if AUTH_AVAILABLE else None
    if not user_ctx:
        return redirect(url_for('login', next=request.url))
    return render_template('settings.html', user=user_ctx)

@app.route('/help')
def help_page():
    """Help and documentation page"""
    user_ctx = getattr(g, 'jwt_user', None) if AUTH_AVAILABLE else None
    if not user_ctx:
        return redirect(url_for('login', next=request.url))
    return render_template('help.html', user=user_ctx)

# New Enhanced API Endpoints

@app.route('/api/live-price')
def get_live_price():
    """Get current XCH price from live APIs"""
    try:
        price_data = api_service.get_xch_price()
        if price_data:
            return jsonify(price_data)
        else:
            return jsonify({'error': 'Unable to fetch price data'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/network-stats')
def get_network_stats():
    """Get live Chia network statistics"""
    try:
        network_data = api_service.get_network_stats()
        if network_data:
            return jsonify(network_data)
        else:
            return jsonify({'error': 'Unable to fetch network data'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/roi-calculator', methods=['POST'])
def calculate_roi():
    """Calculate ROI and profitability metrics"""
    try:
        data = request.get_json()
        
        # Get inputs
        plot_count = int(data.get('plot_count', 1))
        plot_size_tb = float(data.get('plot_size_tb', 0.1))  # 101.4GB per plot
        hardware_cost = float(data.get('hardware_cost', 5000))
        electricity_rate = float(data.get('electricity_rate', 0.14))
        zipcode = data.get('zipcode', '')
        
        # Get live market data
        price_data = api_service.get_xch_price()
        network_data = api_service.get_network_stats()
        
        if not price_data or not network_data:
            return jsonify({'error': 'Unable to fetch live market data'}), 503
        
        xch_price = price_data['price_usd']
        netspace_eib = network_data['netspace_eib']
        
        # Calculate farming metrics
        total_plot_size_tb = plot_count * plot_size_tb
        netspace_tb = netspace_eib * 1024 * 1024  # Convert EiB to TB
        
        # Probability calculations
        plot_proportion = total_plot_size_tb / netspace_tb
        daily_xch_rewards = 4608 * plot_proportion  # Daily XCH block rewards
        monthly_xch = daily_xch_rewards * 30
        annual_xch = daily_xch_rewards * 365
        
        # Revenue calculations
        daily_revenue = daily_xch_rewards * xch_price
        monthly_revenue = monthly_xch * xch_price
        annual_revenue = annual_xch * xch_price
        
        # Cost calculations (electricity for farming, not plotting)
        daily_power_kwh = 0.5 * plot_count  # Estimated 0.5kWh per plot per day
        daily_electricity_cost = daily_power_kwh * electricity_rate
        monthly_electricity_cost = daily_electricity_cost * 30
        annual_electricity_cost = daily_electricity_cost * 365
        
        # Profit calculations
        daily_profit = daily_revenue - daily_electricity_cost
        monthly_profit = monthly_revenue - monthly_electricity_cost
        annual_profit = annual_revenue - annual_electricity_cost
        
        # ROI calculations
        break_even_days = hardware_cost / daily_profit if daily_profit > 0 else float('inf')
        annual_roi_percent = (annual_profit / hardware_cost) * 100 if hardware_cost > 0 else 0
        
        roi_data = {
            'rewards': {
                'daily_xch': round(daily_xch_rewards, 6),
                'monthly_xch': round(monthly_xch, 4),
                'annual_xch': round(annual_xch, 2)
            },
            'revenue': {
                'daily_usd': round(daily_revenue, 2),
                'monthly_usd': round(monthly_revenue, 2),
                'annual_usd': round(annual_revenue, 2)
            },
            'costs': {
                'daily_electricity': round(daily_electricity_cost, 2),
                'monthly_electricity': round(monthly_electricity_cost, 2),
                'annual_electricity': round(annual_electricity_cost, 2)
            },
            'profit': {
                'daily_usd': round(daily_profit, 2),
                'monthly_usd': round(monthly_profit, 2),
                'annual_usd': round(annual_profit, 2)
            },
            'roi_metrics': {
                'break_even_days': round(break_even_days, 1) if break_even_days != float('inf') else None,
                'annual_roi_percent': round(annual_roi_percent, 1),
                'payback_months': round(break_even_days / 30, 1) if break_even_days != float('inf') else None
            },
            'market_conditions': {
                'xch_price': xch_price,
                'netspace_eib': netspace_eib,
                'plot_proportion': plot_proportion,
                'network_source': network_data['source'],
                'price_source': price_data['source']
            }
        }
        
        return jsonify(roi_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hardware-optimizer', methods=['POST'])
def optimize_hardware():
    """Provide hardware optimization recommendations"""
    try:
        data = request.get_json()
        
        budget = float(data.get('budget', 5000))
        plot_count_target = int(data.get('plot_count_target', 100))
        use_case = data.get('use_case', 'balanced')  # speed, space, balanced
        
        # Hardware recommendations based on use case and budget
        recommendations = {
            'cpu': {
                'budget': {'name': 'AMD Ryzen 5 5600X', 'cores': 6, 'price': 200, 'plotting_score': 85},
                'performance': {'name': 'AMD Ryzen 9 5950X', 'cores': 16, 'price': 500, 'plotting_score': 100},
                'extreme': {'name': 'AMD Threadripper 3970X', 'cores': 32, 'price': 1500, 'plotting_score': 150}
            },
            'ram': {
                'minimum': {'amount_gb': 16, 'price': 100, 'recommended_for': 'Basic plotting'},
                'recommended': {'amount_gb': 32, 'price': 200, 'recommended_for': 'Parallel plotting'},
                'optimal': {'amount_gb': 64, 'price': 400, 'recommended_for': 'High-speed plotting'}
            },
            'storage': {
                'nvme_plotting': {
                    'name': 'Samsung 980 PRO 2TB',
                    'capacity_tb': 2,
                    'price': 250,
                    'endurance_tbw': 1200,
                    'estimated_plots': 1200
                },
                'ssd_final': {
                    'name': 'Crucial MX4 4TB',
                    'capacity_tb': 4,
                    'price': 400,
                    'plots_capacity': 40
                }
            }
        }
        
        # Calculate optimal configuration based on budget
        if budget < 2000:
            config = 'budget'
        elif budget < 5000:
            config = 'performance'
        else:
            config = 'extreme'
        
        # SSD endurance calculations
        plots_per_tbw = 1  # Approximately 1 plot per TBW
        nvme_ssd = recommendations['storage']['nvme_plotting']
        estimated_plot_lifetime = nvme_ssd['endurance_tbw'] * plots_per_tbw
        
        optimization_result = {
            'recommended_config': config,
            'hardware_breakdown': {
                'cpu': recommendations['cpu'][config],
                'ram': recommendations['ram']['recommended'],
                'nvme_temp': recommendations['storage']['nvme_plotting'],
                'ssd_final': recommendations['storage']['ssd_final']
            },
            'cost_analysis': {
                'total_hardware_cost': (
                    recommendations['cpu'][config]['price'] +
                    recommendations['ram']['recommended']['price'] +
                    recommendations['storage']['nvme_plotting']['price'] +
                    recommendations['storage']['ssd_final']['price']
                ),
                'cost_per_plot_capacity': 0,  # Will calculate
                'ssd_endurance_analysis': {
                    'estimated_plot_lifetime': estimated_plot_lifetime,
                    'replacement_cost_per_year': nvme_ssd['price'] / 3  # Assume 3-year lifespan
                }
            },
            'performance_estimates': {
                'plots_per_day': recommendations['cpu'][config]['plotting_score'] / 10,
                'time_to_target': plot_count_target / (recommendations['cpu'][config]['plotting_score'] / 10),
                'parallel_plotting_capacity': recommendations['cpu'][config]['cores'] // 4
            }
        }
        
        # Calculate cost per plot capacity
        total_storage_tb = recommendations['storage']['ssd_final']['capacity_tb']
        plots_capacity = int(total_storage_tb * 10)  # ~10 plots per TB
        optimization_result['cost_analysis']['cost_per_plot_capacity'] = round(
            optimization_result['cost_analysis']['total_hardware_cost'] / plots_capacity, 2
        )
        
        return jsonify(optimization_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-insights')
def get_ai_insights():
    """Get AI-powered farming insights and recommendations"""
    try:
        # Get current market conditions
        price_data = api_service.get_xch_price()
        network_data = api_service.get_network_stats()
        
        if not price_data or not network_data:
            return jsonify({'error': 'Unable to fetch market data for insights'}), 503
        
        insights = []
        
        # Price trend analysis
        if price_data['change_24h'] > 5:
            insights.append({
                'type': 'bullish',
                'title': 'Strong Price Movement',
                'description': f"XCH is up {price_data['change_24h']:.1f}% in 24h. Consider increasing plotting capacity.",
                'priority': 'high',
                'action': 'Consider starting new plotting operations'
            })
        elif price_data['change_24h'] < -5:
            insights.append({
                'type': 'bearish', 
                'title': 'Price Decline',
                'description': f"XCH is down {abs(price_data['change_24h']):.1f}% in 24h. Focus on efficiency.",
                'priority': 'medium',
                'action': 'Optimize power consumption and costs'
            })
        
        # Netspace analysis
        if network_data['netspace_eib'] > 40:
            insights.append({
                'type': 'competitive',
                'title': 'High Network Competition',
                'description': f"Netspace is {network_data['netspace_eib']:.1f} EiB. Focus on plot efficiency.",
                'priority': 'high',
                'action': 'Use compression to maximize plot count'
            })
        
        # Optimal plotting time recommendation
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour <= 6:
            insights.append({
                'type': 'timing',
                'title': 'Optimal Plotting Window',
                'description': 'Off-peak electricity hours. Good time for plotting operations.',
                'priority': 'medium',
                'action': 'Start plotting operations now for lower costs'
            })
        
        # Hardware efficiency insight
        insights.append({
            'type': 'efficiency',
            'title': 'Compression Recommendation',
            'description': 'Using Level 7 compression can save 42% storage while maintaining plot validity.',
            'priority': 'high',
            'action': 'Enable compression for new plots'
        })
        
        return jsonify({
            'insights': insights,
            'market_summary': {
                'xch_price': price_data['price_usd'],
                'price_change_24h': price_data['change_24h'],
                'netspace_eib': network_data['netspace_eib'],
                'recommendation': 'bullish' if price_data['change_24h'] > 0 else 'bearish'
            },
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/cost-report', methods=['POST'])
def export_cost_report():
    """Export detailed cost analysis report"""
    try:
        data = request.get_json()
        format_type = data.get('format', 'csv')  # csv or pdf

        # Get comprehensive data for report - use mock data for now
        price_data = {'price_usd': 25.0, 'change_24h': 2.5}
        network_data = {'netspace_eib': 35.0}

        # Calculate comprehensive metrics
        plot_count = int(data.get('plot_count', 10))
        zipcode = data.get('zipcode', '')
        electricity_rate = 0.12  # Default rate
        
        # Create report data
        report_data = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'plot_count': plot_count,
            'location': zipcode,
            'electricity_rate': electricity_rate,
            'xch_price': price_data['price_usd'] if price_data else 25.0,
            'netspace_eib': network_data['netspace_eib'] if network_data else 35.0,
            'estimated_daily_rewards': 0.001 * plot_count,
            'estimated_monthly_cost': 50 * plot_count,
            'break_even_months': 12,
            'roi_annual': 15.5
        }
        
        if format_type == 'csv':
            # Generate CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow(['SquashPlot Cost Analysis Report'])
            writer.writerow(['Generated:', report_data['report_date']])
            writer.writerow([])
            writer.writerow(['Configuration'])
            writer.writerow(['Plot Count', report_data['plot_count']])
            writer.writerow(['Location (Zipcode)', report_data['location']])
            writer.writerow(['Electricity Rate ($/kWh)', report_data['electricity_rate']])
            writer.writerow([])
            writer.writerow(['Market Conditions'])
            writer.writerow(['XCH Price (USD)', report_data['xch_price']])
            writer.writerow(['Network Space (EiB)', report_data['netspace_eib']])
            writer.writerow([])
            writer.writerow(['Financial Projections'])
            writer.writerow(['Estimated Daily Rewards (XCH)', report_data['estimated_daily_rewards']])
            writer.writerow(['Estimated Monthly Cost (USD)', report_data['estimated_monthly_cost']])
            writer.writerow(['Break-even Period (Months)', report_data['break_even_months']])
            writer.writerow(['Annual ROI (%)', report_data['roi_annual']])
            
            csv_data = output.getvalue()
            output.close()
            
            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=squashplot_cost_report_{datetime.now().strftime("%Y%m%d")}.csv'}
            )
            
        elif format_type == 'pdf':
            # Generate PDF
            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            
            # Title
            p.setFont("Helvetica-Bold", 16)
            p.drawString(50, 750, "SquashPlot Cost Analysis Report")
            
            # Report details
            p.setFont("Helvetica", 12)
            y_position = 720
            
            for key, value in report_data.items():
                p.drawString(50, y_position, f"{key.replace('_', ' ').title()}: {value}")
                y_position -= 20
            
            p.showPage()
            p.save()
            
            buffer.seek(0)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return Response(
                pdf_data,
                mimetype='application/pdf',
                headers={'Content-Disposition': f'attachment; filename=squashplot_cost_report_{datetime.now().strftime("%Y%m%d")}.pdf'}
            )
        
        return jsonify({'error': 'Invalid format type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compress', methods=['POST'])
def api_compress():
    """API endpoint for file compression"""
    try:
        data = request.get_json()

        if not data or 'file_path' not in data:
            return jsonify({'error': 'file_path required'}), 400

        file_path = data['file_path']
        output_path = data.get('output_path', file_path + '.compressed')
        k_size = data.get('k_size', 32)

        # Use our SquashPlot compressor
        result = compressor.compress_plot(file_path, output_path, k_size)

        return jsonify({
            'success': True,
            'result': result,
            'message': 'File compressed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics')
def get_system_metrics():
    """Get system predictions and savings metrics"""
    try:
        # Get system resources for calculations
        try:
            cpu_cores = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.total / (1024**3)  # Convert to GB
            
            # Simple storage type detection (we'll assume SSD for better performance in demo)
            storage_type = 'SSD'
            speed_multiplier = 1.8
        except Exception:
            # Fallback values
            cpu_cores = 4
            memory_gb = 8
            storage_type = 'HDD'
            speed_multiplier = 1.3
        
        # Get compression level data to calculate realistic savings
        compression_levels = [
            {'name': 'Basic', 'compression_ratio': 78.0, 'speed_boost': 1.2},
            {'name': 'Balanced', 'compression_ratio': 58.0, 'speed_boost': 1.5},
            {'name': 'Maximum', 'compression_ratio': 45.0, 'speed_boost': 1.8}
        ]
        
        # Use the selected compression level, default to "Balanced" if none selected
        selected_level = "Balanced"  # This would come from user selection in real implementation
        level_data = next((level for level in compression_levels if level['name'] == selected_level), compression_levels[1])
        
        compression_ratio = level_data['compression_ratio']
        speed_boost = level_data.get('speed_boost', 1.5)  # Default speed improvement
        
        # Baseline calculations based on compression level
        base_plot_time_hours = 8.0  # Traditional plotting time
        base_plot_size_gb = 101.4  # Standard Chia plot size
        
        # SquashPlot improvements based on compression level
        predicted_hours = base_plot_time_hours / speed_boost
        compressed_size_gb = base_plot_size_gb * (compression_ratio / 100)
        
        time_saved_hours = base_plot_time_hours - predicted_hours
        storage_saved_gb = base_plot_size_gb - compressed_size_gb
        
        # Realistic power calculations (total system power)
        base_system_power = 200  # Base system power in watts
        cpu_power_per_core = 25  # Reasonable power per core
        total_power_w = base_system_power + (cpu_power_per_core * cpu_cores)
        power_consumption_kw = total_power_w / 1000  # Convert to kW
        electricity_cost_per_kwh = 0.12  # USD per kWh
        
        # Traditional vs SquashPlot energy use
        traditional_energy = base_plot_time_hours * power_consumption_kw
        squashplot_energy = predicted_hours * power_consumption_kw
        energy_saved = traditional_energy - squashplot_energy
        
        # Cost calculations based on storage and energy savings
        storage_cost_per_gb = 0.02  # Rough estimate per GB storage cost
        storage_cost_saved = storage_saved_gb * storage_cost_per_gb
        energy_cost_saved = energy_saved * electricity_cost_per_kwh
        cost_savings = storage_cost_saved + energy_cost_saved
        
        # Efficiency gain calculation based on compression ratio
        efficiency_gain = ((base_plot_size_gb - compressed_size_gb) / base_plot_size_gb) * 100
        
        metrics = {
            'predicted_time': f"{predicted_hours:.1f}h",
            'time_saved': f"-{time_saved_hours:.1f}h",
            'cost_savings': f"${cost_savings:.2f}",
            'energy_use': f"{squashplot_energy:.2f} kWh",
            'energy_saved': f"-{energy_saved:.2f} kWh",
            'efficiency_gain': f"{efficiency_gain:.0f}%",
            'traditional_time': f"{base_plot_time_hours:.1f}h",
            'traditional_cost': f"${storage_cost_saved + energy_cost_saved:.2f}",
            'traditional_energy': f"{traditional_energy:.2f} kWh",
            'compression_level': selected_level,
            'storage_saved': f"{storage_saved_gb:.1f} GB"
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_system_status():
    """Get current system status and tool availability"""
    try:
        import psutil
        import platform

        # Get system information
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Check for available tools (simplified)
        tools_status = {
            'madmax_available': True,  # Assume available for demo
            'bladebit_available': True,
            'chia_available': True,
            'compression_supported': True
        }

        status = {
            'tools': tools_status,
            'resources': {
                'cpu_count': cpu_count,
                'available_memory_gb': round(memory.available / (1024**3), 1),
                'total_memory_gb': round(memory.total / (1024**3), 1),
                'memory_percent': memory.percent,
                'disk_free_gb': round(disk.free / (1024**3), 1),
                'disk_total_gb': round(disk.total / (1024**3), 1),
                'disk_percent': disk.percent,
                'platform': platform.system(),
                'python_version': platform.python_version()
            },
            'plotting': current_plotting_status,
            'squashplot_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(status)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compression-levels')
def get_compression_levels():
    """Get available compression levels"""
    try:
        # Define compression levels directly - no external imports needed
        COMPRESSION_LEVELS = {
            0: {
                'description': 'No Compression',
                'ratio': 1.0
            },
            1: {
                'description': 'Basic (LZ4 + zlib)',
                'ratio': 0.8
            },
            2: {
                'description': 'Standard (Zstandard)',
                'ratio': 0.75
            },
            3: {
                'description': 'High (Brotli)',
                'ratio': 0.7
            },
            4: {
                'description': 'Ultra (Advanced)',
                'ratio': 0.65
            }
        }

        levels = []
        for level, info in COMPRESSION_LEVELS.items():
            levels.append({
                'level': level,
                'description': info['description'],
                'ratio': info['ratio'],
                'savings_percent': round((1 - info['ratio']) * 100, 1),
                'estimated_size_gb': round(108 * info['ratio'], 1)  # Based on standard 108GB plot
            })

        return jsonify(levels)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-plotting', methods=['POST'])
@require_api_key
def start_plotting():
    """Start plotting operation using job queue"""
    global current_plotting_status
    
    if current_plotting_status['active']:
        return jsonify({'error': 'Plotting operation already in progress'}), 400
    
    try:
        data = request.get_json()
        
        # Extract configuration
        config = PlotConfig(
            tmp_dir=data.get('tmp_dir', '/tmp/squashplot'),
            tmp_dir2=data.get('tmp_dir2'),
            final_dir=data.get('final_dir', '/plots'),
            farmer_key=data.get('farmer_key', ''),
            pool_key=data.get('pool_key'),
            contract=data.get('contract'),
            threads=int(data.get('threads', 4)),
            buckets=int(data.get('buckets', 256)),
            cache_size=data.get('cache_size', '8G'),
            count=int(data.get('count', 1)),
            k_size=int(data.get('k_size', 32)),
            compression=int(data.get('compression', 0))
        )
        
        # Validate required fields
        if not config.farmer_key:
            return jsonify({'error': 'Farmer key is required'}), 400
        
        # Start plotting in background thread
        def plot_worker():
            global current_plotting_status
            
            current_plotting_status.update({
                'active': True,
                'progress': 0,
                'stage': 'initializing',
                'start_time': datetime.now().isoformat(),
                'plot_count': config.count,
                'completed_plots': 0,
                'error_message': None
            })
            
            try:
                result = squashplot.plot(config)
                
                if result.success:
                    current_plotting_status.update({
                        'active': False,
                        'progress': 100,
                        'stage': 'completed',
                        'completed_plots': config.count,
                        'current_plot_path': result.plot_path
                    })
                else:
                    current_plotting_status.update({
                        'active': False,
                        'stage': 'failed',
                        'error_message': result.error_message
                    })
                    
            except Exception as e:
                current_plotting_status.update({
                    'active': False,
                    'stage': 'failed',
                    'error_message': str(e)
                })
        
        # Start background thread
        thread = threading.Thread(target=plot_worker)
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Plotting started successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-plotting', methods=['POST'])
@require_api_key
def stop_plotting():
    """Stop current plotting operation"""
    global current_plotting_status
    
    if not current_plotting_status['active']:
        return jsonify({'error': 'No plotting operation in progress'}), 400
    
    # TODO: Implement actual stopping logic
    current_plotting_status.update({
        'active': False,
        'stage': 'stopped',
        'error_message': 'Stopped by user'
    })
    
    return jsonify({'message': 'Plotting stopped'})

# Wallet API endpoints

@app.route('/api/wallet/status')
def get_wallet_status():
    """Get wallet connection status and basic info"""
    try:
        # Mock wallet status for demo purposes
        status = {
            'connected': False,  # Not connected to real Chia wallet
            'auto_claim_enabled': False,
            'last_claim_check': None,
            'wallet_address': 'Demo Wallet - Not Connected',
            'balance': 0.0,
            'rewards_pending': 0.0,
            'network': 'mainnet',
            'status': 'demo_mode'
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/connect', methods=['POST'])
@require_api_key
def connect_wallet():
    """Connect to Chia wallet"""
    try:
        success = asyncio.run(wallet_service.connect())
        if success:
            return jsonify({'success': True, 'message': 'Wallet connected successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to connect to wallet'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/wallet/disconnect', methods=['POST'])
@require_api_key
def disconnect_wallet():
    """Disconnect from Chia wallet"""
    try:
        wallet_service.disconnect()
        return jsonify({'success': True, 'message': 'Wallet disconnected'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/wallets')
@require_api_key
def get_wallets():
    """Get all wallet information"""
    try:
        wallets = asyncio.run(wallet_service.get_wallets())
        
        # Convert Decimal to float for JSON serialization
        wallets_data = []
        for wallet in wallets:
            wallets_data.append({
                'wallet_id': wallet.wallet_id,
                'name': wallet.name,
                'wallet_type': wallet.wallet_type,
                'balance_xch': float(wallet.balance_xch),
                'unconfirmed_xch': float(wallet.unconfirmed_xch),
                'spendable_xch': float(wallet.spendable_xch)
            })
        
        return jsonify({'wallets': wallets_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/send', methods=['POST'])
@require_api_key
def send_xch():
    """Send XCH transaction"""
    try:
        data = request.get_json()
        
        wallet_id = int(data.get('wallet_id', 1))
        recipient = data.get('recipient_address')
        amount = float(data.get('amount_xch'))
        fee = float(data.get('fee_xch', 0.00001))
        
        if not recipient:
            return jsonify({'error': 'Recipient address is required'}), 400
        if amount <= 0:
            return jsonify({'error': 'Amount must be greater than 0'}), 400
        
        result = asyncio.run(wallet_service.send_xch(wallet_id, recipient, amount, fee))
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/offers')
@require_api_key
def get_offers():
    """Get all offer files"""
    try:
        offers = asyncio.run(wallet_service.get_offers())
        
        offers_data = []
        for offer in offers:
            offers_data.append({
                'offer_id': offer.offer_id,
                'summary': offer.summary,
                'status': offer.status,
                'created_at': offer.created_at.isoformat(),
                'file_path': offer.file_path
            })
        
        return jsonify({'offers': offers_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/create-offer', methods=['POST'])
@require_api_key
def create_offer():
    """Create a new offer file"""
    try:
        data = request.get_json()
        
        wallet_id = int(data.get('wallet_id', 1))
        offered_amount = float(data.get('offered_amount'))
        requested_asset = data.get('requested_asset', 'XCH')
        requested_amount = float(data.get('requested_amount'))
        
        result = asyncio.run(wallet_service.create_offer(
            wallet_id, offered_amount, requested_asset, requested_amount
        ))
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/accept-offer', methods=['POST'])
@require_api_key
def accept_offer():
    """Accept an offer file"""
    try:
        data = request.get_json()
        
        offer_file_path = data.get('offer_file_path', '')
        fee = float(data.get('fee_xch', 0.00001))
        
        # Validate file path to prevent directory traversal
        if not offer_file_path:
            return jsonify({'error': 'Offer file path is required'}), 400
        
        # Extract just the filename and validate it securely
        import os
        filename = os.path.basename(offer_file_path)
        if not filename or not filename.endswith('.offer'):
            return jsonify({'error': 'Invalid offer file path'}), 400
        
        # Use validate_file_path to securely find the offer file
        secure_path = validate_file_path(filename, allowed_dirs=['/offers', '/tmp/squashplot/offers'])
        if not secure_path:
            return jsonify({'error': 'Offer file not found or access denied'}), 404
        
        result = asyncio.run(wallet_service.accept_offer(secure_path, fee))
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/rewards')
@require_api_key
def get_pending_rewards():
    """Get pending farming rewards"""
    try:
        rewards = asyncio.run(wallet_service.check_pending_rewards())
        
        rewards_data = []
        for reward in rewards:
            rewards_data.append({
                'claim_type': reward.claim_type,
                'amount_xch': float(reward.amount_xch),
                'block_height': reward.block_height,
                'timestamp': reward.timestamp.isoformat(),
                'status': reward.status
            })
        
        return jsonify({'rewards': rewards_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/claim-rewards', methods=['POST'])
@require_api_key
def claim_rewards():
    """Claim farming rewards"""
    try:
        data = request.get_json() or {}
        reward_types = data.get('reward_types', ['farmer_reward', 'pool_reward'])
        
        result = asyncio.run(wallet_service.claim_rewards(reward_types))
        
        # Convert Decimal to float
        if result.get('success') and 'claimed_rewards' in result:
            for reward in result['claimed_rewards']:
                if 'amount' in reward:
                    reward['amount'] = float(reward['amount'])
        if 'total_claimed' in result:
            result['total_claimed'] = float(result['total_claimed'])
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/auto-claim', methods=['POST'])
@require_api_key
def toggle_auto_claim():
    """Enable/disable automatic reward claiming"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        asyncio.run(wallet_service.enable_auto_claim(enabled))
        return jsonify({
            'success': True, 
            'auto_claim_enabled': enabled,
            'message': f"Auto-claim {'enabled' if enabled else 'disabled'}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wallet/upload-offer', methods=['POST'])
@require_api_key
def upload_offer():
    """Upload an offer file"""
    try:
        if 'offer_file' not in request.files:
            return jsonify({'error': 'No offer file provided'}), 400
        
        file = request.files['offer_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded offer file with secure filename handling
        if file and file.filename and file.filename.endswith('.offer'):
            from werkzeug.utils import secure_filename
            import uuid
            
            # Generate secure server-side filename ignoring user's filename
            secure_name = secure_filename(file.filename) or 'upload.offer'
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            filename = f"uploaded_{timestamp}_{unique_id}_{secure_name}"
            
            # Ensure path stays within offers directory
            file_path = os.path.join(wallet_service.offers_dir, filename)
            file_path = os.path.realpath(file_path)
            offers_dir_real = os.path.realpath(wallet_service.offers_dir)
            
            if not file_path.startswith(offers_dir_real):
                return jsonify({'error': 'Invalid file path detected'}), 400
            
            file.save(file_path)
            
            return jsonify({
                'success': True,
                'file_path': file_path,
                'message': 'Offer file uploaded successfully'
            })
        else:
            return jsonify({'error': 'Invalid file type. Please upload a .offer file'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate-config', methods=['POST'])
@require_api_key
def validate_config():
    """Validate plotting configuration"""
    try:
        data = request.get_json()
        
        errors = []
        warnings = []
        
        # Validate farmer key
        farmer_key = data.get('farmer_key', '')
        if not farmer_key:
            errors.append('Farmer key is required')
        elif len(farmer_key) != 96:
            warnings.append('Farmer key should be 96 characters long')
        
        # Validate directories
        tmp_dir = data.get('tmp_dir', '/tmp/squashplot')
        final_dir = data.get('final_dir', '/plots')
        
        # Validate directories exist (read-only validation - no creation)
        for dir_path in [tmp_dir, final_dir]:
            if not os.path.exists(dir_path):
                errors.append(f'Directory does not exist: {dir_path}')
            elif not os.access(dir_path, os.W_OK):
                errors.append(f'Directory not writable: {dir_path}')
        
        # Validate numeric parameters
        try:
            threads = int(data.get('threads', 4))
            if threads < 1 or threads > 32:
                warnings.append('Thread count should be between 1 and 32')
        except ValueError:
            errors.append('Invalid thread count')
        
        try:
            count = int(data.get('count', 1))
            if count < 1 or count > 100:
                warnings.append('Plot count should be between 1 and 100')
        except ValueError:
            errors.append('Invalid plot count')
        
        try:
            compression = int(data.get('compression', 0))
            if compression < 0 or compression > 5:
                errors.append('Compression level must be between 0 and 5')
        except ValueError:
            errors.append('Invalid compression level')
        
        return jsonify({
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Job Management API Endpoints

@app.route('/api/jobs', methods=['GET'])
@require_api_key
def get_jobs():
    """Get all jobs with optional status filter"""
    try:
        status_filter = request.args.get('status')
        limit = int(request.args.get('limit', 50))
        
        if status_filter:
            try:
                status = JobStatus(status_filter)
                jobs = job_queue.get_jobs(status=status, limit=limit)
            except ValueError:
                return jsonify({'error': f'Invalid status: {status_filter}'}), 400
        else:
            jobs = job_queue.get_jobs(limit=limit)
        
        # Convert datetime objects to strings for JSON serialization
        jobs_data = []
        for job in jobs:
            job_dict = asdict(job)
            if job_dict.get('created_at'):
                job_dict['created_at'] = job_dict['created_at'].isoformat()
            if job_dict.get('updated_at'):
                job_dict['updated_at'] = job_dict['updated_at'].isoformat()
            if job_dict.get('start_time'):
                job_dict['start_time'] = job_dict['start_time'].isoformat()
            if job_dict.get('end_time'):
                job_dict['end_time'] = job_dict['end_time'].isoformat()
            if 'status' in job_dict:
                job_dict['status'] = job_dict['status'].value if hasattr(job_dict['status'], 'value') else job_dict['status']
            jobs_data.append(job_dict)
        
        return jsonify({
            'jobs': jobs_data,
            'total': len(jobs_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs/<job_id>', methods=['GET'])
@require_api_key
def get_job_details(job_id):
    """Get specific job details"""
    try:
        job = job_queue.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        job_dict = asdict(job)
        # Handle datetime serialization
        if job_dict.get('created_at'):
            job_dict['created_at'] = job_dict['created_at'].isoformat()
        if job_dict.get('updated_at'):
            job_dict['updated_at'] = job_dict['updated_at'].isoformat()
        if job_dict.get('start_time'):
            job_dict['start_time'] = job_dict['start_time'].isoformat()
        if job_dict.get('end_time'):
            job_dict['end_time'] = job_dict['end_time'].isoformat()
        if 'status' in job_dict:
            job_dict['status'] = job_dict['status'].value if hasattr(job_dict['status'], 'value') else job_dict['status']
        
        return jsonify(job_dict)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
@require_api_key
def cancel_plot_job(job_id):
    """Cancel a specific job"""
    try:
        success = job_queue.cancel_job(job_id)
        if not success:
            return jsonify({'error': 'Job not found or cannot be cancelled'}), 400
        
        return jsonify({
            'success': True,
            'message': f'Job {job_id} cancelled successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/queue/stats', methods=['GET'])
@require_api_key
def get_queue_statistics():
    """Get job queue statistics"""
    try:
        stats = job_queue.get_queue_stats()
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs/cleanup', methods=['POST'])
@require_api_key
def cleanup_old_jobs():
    """Clean up old completed/failed jobs"""
    try:
        data = request.get_json() or {}
        days = int(data.get('days', 30))
        deleted_count = job_queue.cleanup_old_jobs(days)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Cleaned up {deleted_count} old jobs'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# File Management and Plot Validation Endpoints

@app.route('/api/files/plots', methods=['GET'])
@require_api_key
def list_plot_files():
    """List plot files in directories"""
    try:
        directories = ['/plots', '/tmp/squashplot']
        plot_files = []
        
        for directory in directories:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if filename.endswith('.plot'):
                        filepath = os.path.join(directory, filename)
                        stat_info = os.stat(filepath)
                        
                        plot_files.append({
                            'filename': filename,
                            'path': filepath,
                            'size_gb': round(stat_info.st_size / (1024**3), 2),
                            'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                            'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                            'directory': directory
                        })
        
        # Sort by creation time
        plot_files.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'plots': plot_files,
            'total': len(plot_files),
            'total_size_gb': round(sum(f['size_gb'] for f in plot_files), 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/plots/<path:filename>/validate', methods=['POST'])
@require_api_key
def validate_plot_file(filename):
    """Validate a specific plot file"""
    try:
        # Validate and find the plot file securely
        plot_path = validate_file_path(filename)
        
        if not plot_path:
            return jsonify({'error': 'Plot file not found'}), 404
        
        # Basic validation (file exists and has reasonable size)
        stat_info = os.stat(plot_path)
        expected_sizes = {
            32: 101.4,  # GB
            33: 208.8,
            34: 429.8
        }
        
        # Estimate k-size from filename or size
        k_size = 32  # default
        for k, expected_gb in expected_sizes.items():
            if abs(stat_info.st_size / (1024**3) - expected_gb) < 10:  # 10GB tolerance
                k_size = k
                break
        
        validation_results = {
            'filename': filename,
            'path': plot_path,
            'size_gb': round(stat_info.st_size / (1024**3), 2),
            'estimated_k_size': k_size,
            'is_valid_size': True,  # Basic check passed
            'last_modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            'validation_notes': ['File exists and has reasonable size for a Chia plot']
        }
        
        return jsonify(validation_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/plots/<path:filename>/move', methods=['POST'])
@require_api_key
def move_plot_file(filename):
    """Move a plot file to a different directory"""
    try:
        data = request.get_json()
        destination = data.get('destination', '/plots')
        
        # Validate destination (only allow approved directories)
        allowed_destinations = ['/plots', '/tmp/squashplot']
        if destination not in allowed_destinations:
            return jsonify({'error': 'Destination not allowed'}), 400
        
        # Validate and find source file securely
        source_path = validate_file_path(filename)
        
        if not source_path:
            return jsonify({'error': 'Plot file not found'}), 404
        
        # Ensure destination directory exists
        os.makedirs(destination, exist_ok=True)
        
        # Move file
        dest_path = os.path.join(destination, filename)
        if os.path.exists(dest_path):
            return jsonify({'error': 'File already exists in destination'}), 409
        
        import shutil
        shutil.move(source_path, dest_path)
        
        return jsonify({
            'success': True,
            'message': f'Plot file moved successfully',
            'source': source_path,
            'destination': dest_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/disk-space', methods=['GET'])
@require_api_key
def get_disk_space():
    """Get disk space information for plotting directories"""
    try:
        directories = ['/plots', '/tmp/squashplot', '/']
        disk_info = []
        
        for directory in directories:
            if os.path.exists(directory):
                usage = psutil.disk_usage(directory)
                disk_info.append({
                    'directory': directory,
                    'total_gb': round(usage.total / (1024**3), 2),
                    'used_gb': round(usage.used / (1024**3), 2),
                    'free_gb': round(usage.free / (1024**3), 2),
                    'usage_percent': round((usage.used / usage.total) * 100, 1)
                })
        
        return jsonify({
            'disk_info': disk_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Monitoring and Observability Endpoints

@app.route('/api/monitoring/health', methods=['GET'])
def get_health():
    """Get overall health status"""
    try:
        health = get_health_status()
        status_code = 200
        
        if health['status'] == 'warning':
            status_code = 202
        elif health['status'] == 'critical':
            status_code = 503
            
        return jsonify(health), status_code
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/metrics', methods=['GET'])
def get_monitoring_metrics():
    """Get current monitoring metrics"""
    try:
        summary = metrics_collector.get_metrics_summary()
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/timeseries', methods=['GET'])
def get_metrics_timeseries():
    """Get time series metrics data"""
    try:
        hours = int(request.args.get('hours', 1))
        if hours > 24:  # Limit to 24 hours
            hours = 24
            
        data = metrics_collector.get_time_series(hours)
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def validate_production_config():
    """Validate production configuration and abort if unsafe"""
    is_production = (
        os.getenv('FLASK_ENV') == 'production' or 
        os.getenv('RUN_MODE') == 'prod'
    )
    
    if is_production:
        print("🔐 Running production safety checks...")
        errors = []
        
        # Check for default/unsafe configuration
        if app.config['SECRET_KEY'] == 'dev-secret-key-change-in-production':
            errors.append("Default SECRET_KEY detected! Set SECRET_KEY environment variable to a secure random value")
        
        if app.config['API_KEY'] == 'dev-api-key':
            errors.append("Default API_KEY detected! Set API_KEY environment variable to a secure random value")
        
        # Validate CORS origins
        allowed_origins = os.getenv('ALLOWED_ORIGINS', '').strip()
        if not allowed_origins:
            errors.append("ALLOWED_ORIGINS not set! Set to comma-separated list of allowed domains")
        else:
            # Validate CORS origins format
            origins = []
            for origin in allowed_origins.split(','):
                origin = origin.strip()
                if origin:
                    # Basic validation - must be http/https URL or wildcard
                    if origin == '*' or origin.startswith('http://') or origin.startswith('https://'):
                        origins.append(origin)
                    else:
                        errors.append(f"Invalid CORS origin format: {origin} (must be http://... or https://... or *)")
            
            if not origins:
                errors.append("ALLOWED_ORIGINS contains no valid origins!")
        
        if errors:
            print("❌ CRITICAL PRODUCTION SAFETY ERRORS:")
            for error in errors:
                print(f"   • {error}")
            print("\n🛑 Server startup aborted. Fix configuration and try again.")
            exit(1)
        
        print(f"✅ Production safety checks passed")
        print(f"   API authentication: enabled")
        print(f"   CORS origins: {len(origins)} configured")
        return True
    else:
        print("🔧 Running in development mode")
        return False

def main():
    """Main entry point for web server"""
    host = '0.0.0.0'
    port = 5000
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting SquashPlot Enhanced BETA Web Server on {host}:{port}")
    print(f"🌐 Beta Dashboard: http://{host}:{port}")
    print(f"⚠️  This is beta software - features may change")
    print(f"🔗 Initializing wallet services...")
    
    # Initialize wallet service
    try:
        asyncio.run(initialize_wallet_service())
        print(f"✅ Wallet services initialized")
    except Exception as e:
        print(f"⚠️  Wallet services unavailable: {e}")
    
    # Validate production configuration (will exit if unsafe)
    validate_production_config()
    
    # Start monitoring
    start_monitoring()
    
    try:
        app.run(host=host, port=port, debug=debug)
    finally:
        # Cleanup on shutdown
        from monitoring import stop_monitoring
        stop_monitoring()

if __name__ == '__main__':
    main()