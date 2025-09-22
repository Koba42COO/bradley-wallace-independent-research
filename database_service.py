"""
Enterprise prime aligned compute Platform - Database Service
=====================================================

Comprehensive database service with support for SQLite (development)
and PostgreSQL (production) for data persistence, user management,
and prime aligned compute data storage.

Author: Enterprise prime aligned compute Platform Team
Version: 2.0.0
License: Proprietary
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import contextmanager
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import psycopg2.pool
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    print("Warning: PostgreSQL support not available - using SQLite only")
from cryptography.fernet import Fernet

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration
DB_TYPE = os.getenv('DB_TYPE', 'sqlite')  # 'sqlite' or 'postgresql'
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'consciousness_platform')
DB_USER = os.getenv('DB_USER', 'consciousness_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'consciousness_pass')

# Encryption for sensitive data
ENCRYPTION_KEY = os.getenv('DB_ENCRYPTION_KEY', Fernet.generate_key())
fernet = Fernet(ENCRYPTION_KEY)

class DatabaseService:
    """
    Unified database service supporting SQLite and PostgreSQL
    with automatic schema management and data encryption.
    """

    def __init__(self):
        self.db_type = DB_TYPE
        self.connection_pool = None
        self.sqlite_conn = None

        if self.db_type == 'postgresql':
            self._init_postgresql()
        else:
            self._init_sqlite()

        self._create_tables()
        logger.info(f"Database service initialized with {self.db_type}")

    def _init_sqlite(self):
        """Initialize SQLite database"""
        db_path = os.path.join(os.path.dirname(__file__), 'consciousness_platform.db')
        self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row
        logger.info(f"SQLite database initialized at {db_path}")

    def _init_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        if not POSTGRESQL_AVAILABLE:
            logger.warning("PostgreSQL not available, falling back to SQLite")
            self.db_type = 'sqlite'
            self._init_sqlite()
            return

        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=20,
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            # Fallback to SQLite
            self.db_type = 'sqlite'
            self._init_sqlite()

    @contextmanager
    def get_connection(self):
        """Get database connection (context manager)"""
        if self.db_type == 'postgresql':
            conn = self.connection_pool.getconn()
            try:
                yield conn
            finally:
                self.connection_pool.putconn(conn)
        else:
            yield self.sqlite_conn

    def _create_tables(self):
        """Create database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            if self.db_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id VARCHAR(50) PRIMARY KEY,
                        username VARCHAR(100) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        full_name VARCHAR(255),
                        role VARCHAR(50) DEFAULT 'user',
                        password_hash TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        permissions JSONB DEFAULT '["read"]',
                        api_keys JSONB DEFAULT '{}',
                        profile_data JSONB DEFAULT '{}'
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        full_name TEXT,
                        role TEXT DEFAULT 'user',
                        password_hash TEXT NOT NULL,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        last_login TEXT,
                        permissions TEXT DEFAULT '["read"]',
                        api_keys TEXT DEFAULT '{}',
                        profile_data TEXT DEFAULT '{}'
                    )
                """)

            # prime aligned compute data table
            if self.db_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS consciousness_data (
                        id SERIAL PRIMARY KEY,
                        data_type VARCHAR(50) NOT NULL,
                        data_value JSONB NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        created_by VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS consciousness_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data_type TEXT NOT NULL,
                        data_value TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}',
                        created_by TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        is_active INTEGER DEFAULT 1
                    )
                """)

            # Processing history table
            if self.db_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processing_history (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50),
                        algorithm VARCHAR(100) NOT NULL,
                        input_data JSONB,
                        output_data JSONB,
                        processing_time FLOAT,
                        status VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        error_message TEXT
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        algorithm TEXT NOT NULL,
                        input_data TEXT,
                        output_data TEXT,
                        processing_time REAL,
                        status TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        error_message TEXT
                    )
                """)

            # Sessions table for JWT refresh tokens
            if self.db_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50) NOT NULL,
                        refresh_token_hash VARCHAR(128) UNIQUE NOT NULL,
                        encrypted_token TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        refresh_token_hash TEXT UNIQUE NOT NULL,
                        encrypted_token TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        expires_at TEXT NOT NULL,
                        is_active INTEGER DEFAULT 1
                    )
                """)

            # API keys table
            if self.db_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50) NOT NULL,
                        name VARCHAR(100) NOT NULL,
                        key_hash VARCHAR(128) UNIQUE NOT NULL,
                        permissions JSONB DEFAULT '["read"]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_used TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        key_hash TEXT UNIQUE NOT NULL,
                        permissions TEXT DEFAULT '["read"]',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        last_used TEXT,
                        is_active INTEGER DEFAULT 1
                    )
                """)

            # System metrics table
            if self.db_type == 'postgresql':
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id SERIAL PRIMARY KEY,
                        metric_type VARCHAR(100) NOT NULL,
                        metric_value JSONB NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_type TEXT NOT NULL,
                        metric_value TEXT NOT NULL,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            conn.commit()
            logger.info("Database tables created successfully")

    # User management methods
    def create_user(self, user_id: str, username: str, email: str, full_name: str,
                   password_hash: str, role: str = 'user') -> bool:
        """Create new user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        INSERT INTO users (id, username, email, full_name, password_hash, role)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (user_id, username, email, full_name, password_hash, role))
                else:
                    cursor.execute("""
                        INSERT INTO users (id, username, email, full_name, password_hash, role)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (user_id, username, email, full_name, password_hash, role))

                conn.commit()
                logger.info(f"User {username} created successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to create user {username}: {e}")
                conn.rollback()
                return False

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if self.db_type == 'postgresql':
                cursor.execute("""
                    SELECT id, username, email, full_name, role, password_hash,
                           is_active, created_at, last_login, permissions, api_keys, profile_data
                    FROM users WHERE username = %s
                """, (username,))
            else:
                cursor.execute("""
                    SELECT id, username, email, full_name, role, password_hash,
                           is_active, created_at, last_login, permissions, api_keys, profile_data
                    FROM users WHERE username = ?
                """, (username,))

            row = cursor.fetchone()
            if row:
                user_data = dict(row) if self.db_type == 'postgresql' else dict(row)
                # Parse JSON fields
                user_data['permissions'] = json.loads(user_data['permissions'])
                user_data['api_keys'] = json.loads(user_data['api_keys'])
                user_data['profile_data'] = json.loads(user_data['profile_data'])
                return user_data

        return None

    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """Update user information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Build dynamic update query
            set_parts = []
            values = []

            for key, value in updates.items():
                if key in ['permissions', 'api_keys', 'profile_data']:
                    value = json.dumps(value)
                set_parts.append(f"{key} = %s" if self.db_type == 'postgresql' else f"{key} = ?")
                values.append(value)

            values.append(username)

            query = f"""
                UPDATE users
                SET {', '.join(set_parts)}
                WHERE username = {'%s' if self.db_type == 'postgresql' else '?'}
            """

            try:
                cursor.execute(query, values)
                conn.commit()
                logger.info(f"User {username} updated successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to update user {username}: {e}")
                conn.rollback()
                return False

    def delete_user(self, username: str) -> bool:
        """Delete user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                if self.db_type == 'postgresql':
                    cursor.execute("DELETE FROM users WHERE username = %s", (username,))
                else:
                    cursor.execute("DELETE FROM users WHERE username = ?", (username,))

                conn.commit()
                logger.info(f"User {username} deleted successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to delete user {username}: {e}")
                conn.rollback()
                return False

    def list_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List users with pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if self.db_type == 'postgresql':
                cursor.execute("""
                    SELECT id, username, email, role, is_active, created_at, last_login
                    FROM users
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
            else:
                cursor.execute("""
                    SELECT id, username, email, role, is_active, created_at, last_login
                    FROM users
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

            users = []
            for row in cursor.fetchall():
                user_data = dict(row) if self.db_type == 'postgresql' else dict(row)
                users.append(user_data)

            return users

    # Session management
    def store_refresh_token(self, user_id: str, refresh_token_hash: str, encrypted_token: str,
                           expires_at: datetime) -> bool:
        """Store encrypted refresh token"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        INSERT INTO sessions (user_id, refresh_token_hash, encrypted_token, expires_at)
                        VALUES (%s, %s, %s, %s)
                    """, (user_id, refresh_token_hash, encrypted_token, expires_at))
                else:
                    cursor.execute("""
                        INSERT INTO sessions (user_id, refresh_token_hash, encrypted_token, expires_at)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, refresh_token_hash, encrypted_token, expires_at.isoformat()))

                conn.commit()
                return True

            except Exception as e:
                logger.error(f"Failed to store refresh token: {e}")
                conn.rollback()
                return False

    def get_refresh_token(self, refresh_token_hash: str) -> Optional[Dict[str, Any]]:
        """Get refresh token data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if self.db_type == 'postgresql':
                cursor.execute("""
                    SELECT user_id, encrypted_token, expires_at, is_active
                    FROM sessions
                    WHERE refresh_token_hash = %s AND is_active = TRUE
                """, (refresh_token_hash,))
            else:
                cursor.execute("""
                    SELECT user_id, encrypted_token, expires_at, is_active
                    FROM sessions
                    WHERE refresh_token_hash = ? AND is_active = 1
                """, (refresh_token_hash,))

            row = cursor.fetchone()
            if row:
                return dict(row) if self.db_type == 'postgresql' else dict(row)

        return None

    def revoke_refresh_token(self, user_id: str) -> bool:
        """Revoke all refresh tokens for user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        UPDATE sessions SET is_active = FALSE
                        WHERE user_id = %s
                    """, (user_id,))
                else:
                    cursor.execute("""
                        UPDATE sessions SET is_active = 0
                        WHERE user_id = ?
                    """, (user_id,))

                conn.commit()
                return True

            except Exception as e:
                logger.error(f"Failed to revoke refresh tokens for user {user_id}: {e}")
                conn.rollback()
                return False

    # API key management
    def store_api_key(self, user_id: str, name: str, key_hash: str, permissions: List[str]) -> bool:
        """Store API key"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                permissions_json = json.dumps(permissions)

                if self.db_type == 'postgresql':
                    cursor.execute("""
                        INSERT INTO api_keys (user_id, name, key_hash, permissions)
                        VALUES (%s, %s, %s, %s)
                    """, (user_id, name, key_hash, permissions_json))
                else:
                    cursor.execute("""
                        INSERT INTO api_keys (user_id, name, key_hash, permissions)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, name, key_hash, permissions_json))

                conn.commit()
                return True

            except Exception as e:
                logger.error(f"Failed to store API key: {e}")
                conn.rollback()
                return False

    def get_api_key(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Get API key data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if self.db_type == 'postgresql':
                cursor.execute("""
                    SELECT user_id, name, permissions, created_at, last_used, is_active
                    FROM api_keys
                    WHERE key_hash = %s AND is_active = TRUE
                """, (key_hash,))
            else:
                cursor.execute("""
                    SELECT user_id, name, permissions, created_at, last_used, is_active
                    FROM api_keys
                    WHERE key_hash = ? AND is_active = 1
                """, (key_hash,))

            row = cursor.fetchone()
            if row:
                key_data = dict(row) if self.db_type == 'postgresql' else dict(row)
                key_data['permissions'] = json.loads(key_data['permissions'])
                return key_data

        return None

    def list_user_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """List API keys for user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if self.db_type == 'postgresql':
                cursor.execute("""
                    SELECT name, permissions, created_at, last_used, is_active
                    FROM api_keys
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT name, permissions, created_at, last_used, is_active
                    FROM api_keys
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))

            keys = []
            for row in cursor.fetchall():
                key_data = dict(row) if self.db_type == 'postgresql' else dict(row)
                key_data['permissions'] = json.loads(key_data['permissions'])
                keys.append(key_data)

            return keys

    def revoke_api_key(self, user_id: str, key_name: str) -> bool:
        """Revoke API key"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        UPDATE api_keys SET is_active = FALSE
                        WHERE user_id = %s AND name = %s
                    """, (user_id, key_name))
                else:
                    cursor.execute("""
                        UPDATE api_keys SET is_active = 0
                        WHERE user_id = ? AND name = ?
                    """, (user_id, key_name))

                conn.commit()
                return True

            except Exception as e:
                logger.error(f"Failed to revoke API key {key_name}: {e}")
                conn.rollback()
                return False

    # prime aligned compute data management
    def store_consciousness_data(self, data_type: str, data_value: Dict[str, Any],
                               metadata: Dict[str, Any] = None, created_by: str = None) -> int:
        """Store prime aligned compute data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                data_value_json = json.dumps(data_value)
                metadata_json = json.dumps(metadata or {})

                if self.db_type == 'postgresql':
                    cursor.execute("""
                        INSERT INTO consciousness_data (data_type, data_value, metadata, created_by)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (data_type, data_value_json, metadata_json, created_by))

                    return cursor.fetchone()[0]

                else:
                    cursor.execute("""
                        INSERT INTO consciousness_data (data_type, data_value, metadata, created_by)
                        VALUES (?, ?, ?, ?)
                    """, (data_type, data_value_json, metadata_json, created_by))

                    return cursor.lastrowid

            except Exception as e:
                logger.error(f"Failed to store prime aligned compute data: {e}")
                conn.rollback()
                return -1

    def get_consciousness_data(self, data_type: str = None, limit: int = 100,
                             offset: int = 0) -> List[Dict[str, Any]]:
        """Get prime aligned compute data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if data_type:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        SELECT id, data_type, data_value, metadata, created_by, created_at, updated_at
                        FROM consciousness_data
                        WHERE data_type = %s AND is_active = TRUE
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """, (data_type, limit, offset))
                else:
                    cursor.execute("""
                        SELECT id, data_type, data_value, metadata, created_by, created_at, updated_at
                        FROM consciousness_data
                        WHERE data_type = ? AND is_active = 1
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    """, (data_type, limit, offset))
            else:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        SELECT id, data_type, data_value, metadata, created_by, created_at, updated_at
                        FROM consciousness_data
                        WHERE is_active = TRUE
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """, (limit, offset))
                else:
                    cursor.execute("""
                        SELECT id, data_type, data_value, metadata, created_by, created_at, updated_at
                        FROM consciousness_data
                        WHERE is_active = 1
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    """, (limit, offset))

            data = []
            for row in cursor.fetchall():
                item = dict(row) if self.db_type == 'postgresql' else dict(row)
                item['data_value'] = json.loads(item['data_value'])
                item['metadata'] = json.loads(item['metadata'])
                data.append(item)

            return data

    # Processing history
    def store_processing_history(self, user_id: str, algorithm: str, input_data: Dict[str, Any],
                               output_data: Dict[str, Any], processing_time: float, status: str,
                               error_message: str = None) -> bool:
        """Store processing history"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                input_json = json.dumps(input_data) if input_data else None
                output_json = json.dumps(output_data) if output_data else None

                if self.db_type == 'postgresql':
                    cursor.execute("""
                        INSERT INTO processing_history
                        (user_id, algorithm, input_data, output_data, processing_time, status, error_message)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (user_id, algorithm, input_json, output_json, processing_time, status, error_message))
                else:
                    cursor.execute("""
                        INSERT INTO processing_history
                        (user_id, algorithm, input_data, output_data, processing_time, status, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, algorithm, input_json, output_json, processing_time, status, error_message))

                conn.commit()
                return True

            except Exception as e:
                logger.error(f"Failed to store processing history: {e}")
                conn.rollback()
                return False

    def get_processing_history(self, user_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get processing history"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if user_id:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        SELECT id, user_id, algorithm, input_data, output_data,
                               processing_time, status, created_at, error_message
                        FROM processing_history
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, limit))
                else:
                    cursor.execute("""
                        SELECT id, user_id, algorithm, input_data, output_data,
                               processing_time, status, created_at, error_message
                        FROM processing_history
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (user_id, limit))
            else:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        SELECT id, user_id, algorithm, input_data, output_data,
                               processing_time, status, created_at, error_message
                        FROM processing_history
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))
                else:
                    cursor.execute("""
                        SELECT id, user_id, algorithm, input_data, output_data,
                               processing_time, status, created_at, error_message
                        FROM processing_history
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (limit,))

            history = []
            for row in cursor.fetchall():
                item = dict(row) if self.db_type == 'postgresql' else dict(row)
                if item['input_data']:
                    item['input_data'] = json.loads(item['input_data'])
                if item['output_data']:
                    item['output_data'] = json.loads(item['output_data'])
                history.append(item)

            return history

    # System metrics
    def store_system_metric(self, metric_type: str, metric_value: Dict[str, Any]) -> bool:
        """Store system metric"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                value_json = json.dumps(metric_value)

                if self.db_type == 'postgresql':
                    cursor.execute("""
                        INSERT INTO system_metrics (metric_type, metric_value)
                        VALUES (%s, %s)
                    """, (metric_type, value_json))
                else:
                    cursor.execute("""
                        INSERT INTO system_metrics (metric_type, metric_value)
                        VALUES (?, ?)
                    """, (metric_type, value_json))

                conn.commit()
                return True

            except Exception as e:
                logger.error(f"Failed to store system metric: {e}")
                conn.rollback()
                return False

    def get_system_metrics(self, metric_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get system metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if metric_type:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        SELECT id, metric_type, metric_value, timestamp
                        FROM system_metrics
                        WHERE metric_type = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (metric_type, limit))
                else:
                    cursor.execute("""
                        SELECT id, metric_type, metric_value, timestamp
                        FROM system_metrics
                        WHERE metric_type = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (metric_type, limit))
            else:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        SELECT id, metric_type, metric_value, timestamp
                        FROM system_metrics
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                else:
                    cursor.execute("""
                        SELECT id, metric_type, metric_value, timestamp
                        FROM system_metrics
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (limit,))

            metrics = []
            for row in cursor.fetchall():
                item = dict(row) if self.db_type == 'postgresql' else dict(row)
                item['metric_value'] = json.loads(item['metric_value'])
                metrics.append(item)

            return metrics

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                if self.db_type == 'postgresql':
                    cursor.execute("""
                        DELETE FROM sessions
                        WHERE expires_at < CURRENT_TIMESTAMP OR is_active = FALSE
                    """)
                else:
                    cursor.execute("""
                        DELETE FROM sessions
                        WHERE expires_at < datetime('now') OR is_active = 0
                    """)

                deleted_count = cursor.rowcount
                conn.commit()
                logger.info(f"Cleaned up {deleted_count} expired sessions")
                return deleted_count

            except Exception as e:
                logger.error(f"Failed to cleanup expired sessions: {e}")
                conn.rollback()
                return 0

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {
                "database_type": self.db_type,
                "tables": {}
            }

            # Get table row counts
            tables = ["users", "consciousness_data", "processing_history", "sessions", "api_keys", "system_metrics"]

            for table in tables:
                try:
                    if self.db_type == 'postgresql':
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    else:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")

                    count = cursor.fetchone()[0]
                    stats["tables"][table] = count

                except Exception as e:
                    logger.warning(f"Failed to get stats for table {table}: {e}")
                    stats["tables"][table] = 0

            return stats

# Global database service instance
database_service = DatabaseService()

if __name__ == "__main__":
    # Test the database service
    print("🧪 Testing Database Service...")

    # Test user creation
    success = database_service.create_user(
        "test-user-001",
        "testuser",
        "test@example.com",
        "Test User",
        "hashed_password_123",
        "researcher"
    )
    print(f"✅ User creation: {'SUCCESS' if success else 'FAILED'}")

    # Test user retrieval
    user = database_service.get_user_by_username("testuser")
    if user:
        print(f"✅ User retrieval: {user['username']} ({user['role']})")
    else:
        print("❌ User retrieval failed")

    # Test prime aligned compute data storage
    data_id = database_service.store_consciousness_data(
        "test_data",
        {"test": "value", "number": 42},
        {"source": "test", "quality": "high"},
        "testuser"
    )
    print(f"✅ Data storage: {'SUCCESS' if data_id > 0 else 'FAILED'} (ID: {data_id})")

    # Test data retrieval
    data = database_service.get_consciousness_data("test_data", limit=5)
    print(f"✅ Data retrieval: {len(data)} records found")

    # Test database stats
    stats = database_service.get_database_stats()
    print(f"✅ Database stats: {stats['database_type']} with {len(stats['tables'])} tables")

    print("🎉 Database service test completed!")
