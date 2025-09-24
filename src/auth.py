"""
Authentication system for SquashPlot Enhanced
Integrates with Replit Auth for wallet-based sign-in
"""

import jwt
import os
import uuid
import time
from datetime import datetime
from functools import wraps
from urllib.parse import urlencode

from flask import g, session, redirect, request, render_template, url_for, current_app
from flask_dance.consumer import (
    OAuth2ConsumerBlueprint,
    oauth_authorized,
    oauth_error,
)
from flask_dance.consumer.storage import BaseStorage
from flask_login import LoginManager, login_user, logout_user, current_user
from oauthlib.oauth2.rfc6749.errors import InvalidGrantError
from sqlalchemy.exc import NoResultFound
from werkzeug.local import LocalProxy

try:
    from models import db, OAuth, User
except ImportError:
    # Fallback for development
    db = None
    OAuth = None
    User = None

# Initialize login manager
login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

class UserSessionStorage(BaseStorage):
    """Custom storage for OAuth tokens"""

    def get(self, blueprint):
        try:
            token = db.session.query(OAuth).filter_by(
                user_id=current_user.get_id(),
                browser_session_key=g.browser_session_key,
                provider=blueprint.name,
            ).one().token
        except NoResultFound:
            token = None
        return token

    def set(self, blueprint, token):
        db.session.query(OAuth).filter_by(
            user_id=current_user.get_id(),
            browser_session_key=g.browser_session_key,
            provider=blueprint.name,
        ).delete()
        new_model = OAuth()
        new_model.user_id = current_user.get_id()
        new_model.browser_session_key = g.browser_session_key
        new_model.provider = blueprint.name
        new_model.token = token
        db.session.add(new_model)
        db.session.commit()

    def delete(self, blueprint):
        db.session.query(OAuth).filter_by(
            user_id=current_user.get_id(),
            browser_session_key=g.browser_session_key,
            provider=blueprint.name).delete()
        db.session.commit()

def make_replit_blueprint():
    """Create Replit OAuth blueprint"""
    try:
        repl_id = os.environ['REPL_ID']
    except KeyError:
        # Development fallback - return None if REPL_ID not available
        return None

    issuer_url = os.environ.get('ISSUER_URL', "https://replit.com/oidc")

    replit_bp = OAuth2ConsumerBlueprint(
        "replit_auth",
        __name__,
        client_id=repl_id,
        client_secret=None,
        base_url=issuer_url,
        authorization_url_params={
            "prompt": "login consent",
        },
        token_url=issuer_url + "/token",
        token_url_params={
            "auth": (),
            "include_client_id": True,
        },
        auto_refresh_url=issuer_url + "/token",
        auto_refresh_kwargs={
            "client_id": repl_id,
        },
        authorization_url=issuer_url + "/auth",
        use_pkce=True,
        code_challenge_method="S256",
        scope=["openid", "profile", "email", "offline_access"],
        storage=UserSessionStorage(),
    )

    @replit_bp.before_app_request
    def set_applocal_session():
        if '_browser_session_key' not in session:
            session['_browser_session_key'] = uuid.uuid4().hex
        session.modified = True
        g.browser_session_key = session['_browser_session_key']
        g.flask_dance_replit = replit_bp.session

    @replit_bp.route("/logout")
    def logout():
        del replit_bp.token
        logout_user()

        end_session_endpoint = issuer_url + "/session/end"
        encoded_params = urlencode({
            "client_id": repl_id,
            "post_logout_redirect_uri": request.url_root,
        })
        logout_url = f"{end_session_endpoint}?{encoded_params}"

        return redirect(logout_url)

    @replit_bp.route("/error")
    def error():
        return render_template("error.html", 
                             error_message="Authentication failed. Please try again."), 403

    return replit_bp

def save_user(user_claims):
    """Save or update user from authentication claims"""
    user = User()
    user.id = user_claims['sub']
    user.email = user_claims.get('email')
    user.first_name = user_claims.get('first_name')
    user.last_name = user_claims.get('last_name')
    user.profile_image_url = user_claims.get('profile_image_url')
    user.last_login = datetime.now()
    
    merged_user = db.session.merge(user)
    db.session.commit()
    return merged_user

@oauth_authorized.connect
def logged_in(blueprint, token):
    """Handle successful OAuth login"""
    # Properly verify JWT token with JWKS
    issuer_url = os.environ.get('ISSUER_URL', "https://replit.com/oidc")
    repl_id = os.environ.get('REPL_ID')
    
    if os.getenv('FLASK_ENV') == 'production' and repl_id:
        # In production, verify signature with proper JWKS
        jwks_url = f"{issuer_url}/jwks"
        jwks_client = jwt.PyJWKClient(jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token['id_token'])
        
        user_claims = jwt.decode(
            token['id_token'],
            signing_key.key,
            algorithms=["RS256"],
            audience=repl_id,
            issuer=issuer_url,
            options={"verify_signature": True, "verify_aud": True, "verify_iss": True}
        )
    else:
        # Development mode - decode with basic validation
        user_claims = jwt.decode(
            token['id_token'], 
            options={"verify_signature": False, "verify_aud": False, "verify_iss": False}
        )
    user = save_user(user_claims)
    login_user(user)
    blueprint.token = token
    next_url = session.pop("next_url", None)
    if next_url is not None:
        return redirect(next_url)

@oauth_error.connect
def handle_error(blueprint, error, error_description=None, error_uri=None):
    """Handle OAuth errors"""
    current_app.logger.error(f"OAuth error: {error} - {error_description}")
    return redirect(url_for('replit_auth.error'))

def require_login(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            session["next_url"] = get_next_navigation_url(request)
            return redirect(url_for('replit_auth.login'))

        # Check if token needs refresh (using expires_at)
        current_time = time.time()
        token_obj = replit.token if hasattr(g, 'flask_dance_replit') and replit.token else None
        
        if token_obj and 'expires_at' in token_obj:
            if current_time >= token_obj['expires_at']:
                try:
                    # Refresh token using OAuth2Session method
                    refresh_token_url = os.environ.get('ISSUER_URL', "https://replit.com/oidc") + "/token"
                    new_token = replit.refresh_token(refresh_token_url)
                    # Persist via blueprint storage
                    blueprint = current_app.blueprints.get('replit_auth')
                    if blueprint:
                        blueprint.token = new_token
                except Exception:
                    # If refresh fails, redirect to login
                    session["next_url"] = get_next_navigation_url(request)
                    return redirect(url_for('replit_auth.login'))

        return f(*args, **kwargs)

    return decorated_function

def get_next_navigation_url(request):
    """Get the URL to redirect to after login"""
    is_navigation_url = request.headers.get(
        'Sec-Fetch-Mode') == 'navigate' and request.headers.get(
            'Sec-Fetch-Dest') == 'document'
    if is_navigation_url:
        return request.url
    return request.referrer or request.url

def init_auth(app):
    """Initialize authentication system"""
    login_manager.init_app(app)
    
    # Only initialize OAuth if REPL_ID is available
    replit_bp = make_replit_blueprint()
    if replit_bp:
        login_manager.login_view = 'replit_auth.login'
        login_manager.login_message = 'Please sign in to access SquashPlot features.'
        app.register_blueprint(replit_bp, url_prefix="/auth")
        app.logger.info("Replit Auth initialized successfully")
    else:
        app.logger.warning("REPL_ID not found - using development authentication mode")
        # Set up development authentication routes
        @app.route('/auth/login')
        def dev_login():
            # Create a mock user for development
            if not db:
                return redirect(url_for('index'))
            
            # Find or create a dev user
            dev_user = User.query.filter_by(id='dev_user').first()
            if not dev_user:
                dev_user = User(
                    id='dev_user',
                    email='dev@squashplot.local',
                    first_name='Development',
                    last_name='User',
                    farm_name='Dev Farm'
                )
                db.session.add(dev_user)
                db.session.commit()
            
            login_user(dev_user)
            next_url = session.pop('next_url', None) or url_for('index')
            return redirect(next_url)
            
        @app.route('/auth/logout')
        def dev_logout():
            logout_user()
            return redirect(url_for('index'))
    
    return replit_bp

# Create proxy for accessing the OAuth session
replit = LocalProxy(lambda: g.flask_dance_replit)