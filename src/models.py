"""
Database models for SquashPlot Enhanced
Includes user authentication and farming data
"""

import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_dance.consumer.storage.sqla import OAuthConsumerMixin
from flask_login import UserMixin
from sqlalchemy import UniqueConstraint

# Initialize SQLAlchemy
db = SQLAlchemy()

# User authentication model (Required for Replit Auth)
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String, primary_key=True)
    email = db.Column(db.String, unique=True, nullable=True)
    first_name = db.Column(db.String, nullable=True)
    last_name = db.Column(db.String, nullable=True)
    profile_image_url = db.Column(db.String, nullable=True)
    
    # Farming profile data
    farm_name = db.Column(db.String, nullable=True)
    total_plots = db.Column(db.Integer, default=0)
    total_capacity_tb = db.Column(db.Float, default=0.0)
    wallet_address = db.Column(db.String, nullable=True)
    preferred_pool = db.Column(db.String, nullable=True)
    hardware_config = db.Column(db.Text, nullable=True)  # JSON string
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    last_login = db.Column(db.DateTime, nullable=True)

    # Relationships
    plot_jobs = db.relationship('PlotJob', backref='user', lazy=True)
    chat_messages = db.relationship('ChatMessage', backref='user', lazy=True)

    @property
    def display_name(self):
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.email:
            return self.email.split('@')[0]
        return f"User {self.id[:8]}"

# OAuth token storage (Required for Replit Auth)
class OAuth(OAuthConsumerMixin, db.Model):
    user_id = db.Column(db.String, db.ForeignKey(User.id))
    browser_session_key = db.Column(db.String, nullable=False)
    user = db.relationship(User)

    __table_args__ = (UniqueConstraint(
        'user_id',
        'browser_session_key',
        'provider',
        name='uq_user_browser_session_key_provider',
    ),)

# Plot job tracking
class PlotJob(db.Model):
    __tablename__ = 'plot_jobs'
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    
    # Job details
    plot_size = db.Column(db.String, nullable=False)  # k32, k33, etc.
    plot_count = db.Column(db.Integer, default=1)
    compression_level = db.Column(db.String, default='basic')
    status = db.Column(db.String, default='pending')
    progress = db.Column(db.Float, default=0.0)
    
    # Paths and configuration
    plot_dir = db.Column(db.String, nullable=False)
    temp_dir = db.Column(db.String, nullable=False)
    config_json = db.Column(db.Text, nullable=True)  # JSON string
    
    # Performance metrics
    estimated_time = db.Column(db.Integer, nullable=True)  # seconds
    actual_time = db.Column(db.Integer, nullable=True)  # seconds
    compression_ratio = db.Column(db.Float, nullable=True)
    power_usage_kwh = db.Column(db.Float, nullable=True)
    cost_usd = db.Column(db.Float, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)

# Chat messages for SquashChat
class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    
    # Message content
    message = db.Column(db.Text, nullable=False)
    channel = db.Column(db.String, default='general')  # general, pools, trading, etc.
    reply_to = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=True)
    
    # Message metadata
    timestamp = db.Column(db.DateTime, default=datetime.now)
    edited_at = db.Column(db.DateTime, nullable=True)
    is_system = db.Column(db.Boolean, default=False)
    
    # Replies relationship
    replies = db.relationship('ChatMessage', backref=db.backref('parent', remote_side=[id]))

# Pool coordination
class FarmingPool(db.Model):
    __tablename__ = 'farming_pools'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Pool details
    pool_name = db.Column(db.String, nullable=False)
    pool_url = db.Column(db.String, nullable=False)
    launcher_id = db.Column(db.String, unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Pool stats
    total_farmers = db.Column(db.Integer, default=0)
    total_plots = db.Column(db.Integer, default=0)
    pool_space_tb = db.Column(db.Float, default=0.0)
    fee_percent = db.Column(db.Float, default=1.0)
    min_payout_xch = db.Column(db.Float, default=0.01)
    
    # Performance metrics
    average_effort = db.Column(db.Float, nullable=True)
    blocks_found_24h = db.Column(db.Integer, default=0)
    estimated_time_to_win = db.Column(db.Integer, nullable=True)  # hours
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

# Pool membership tracking
class PoolMembership(db.Model):
    __tablename__ = 'pool_memberships'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    pool_id = db.Column(db.Integer, db.ForeignKey('farming_pools.id'), nullable=False)
    
    # Membership details
    farmer_id = db.Column(db.String, nullable=True)
    payout_address = db.Column(db.String, nullable=True)
    plot_count = db.Column(db.Integer, default=0)
    effective_space_tb = db.Column(db.Float, default=0.0)
    
    # Performance tracking
    points_24h = db.Column(db.Integer, default=0)
    rewards_24h_xch = db.Column(db.Float, default=0.0)
    total_rewards_xch = db.Column(db.Float, default=0.0)
    last_payout = db.Column(db.DateTime, nullable=True)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    joined_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    user = db.relationship('User', backref='pool_memberships')
    pool = db.relationship('FarmingPool', backref='members')