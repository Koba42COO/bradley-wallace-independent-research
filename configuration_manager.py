#!/usr/bin/env python3
"""
Configuration Management System
===============================
Comprehensive configuration management for all chAIos platform services
Handles environment variables, service configs, database settings, and deployment parameters.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class ServiceConfig:
    """Configuration for individual services"""
    name: str
    enabled: bool = True
    port: Optional[int] = None
    host: str = "localhost"
    workers: int = 1
    timeout: int = 30
    debug: bool = False
    log_level: str = "INFO"
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"  # sqlite, postgresql, mongodb
    host: str = "localhost"
    port: Optional[int] = None
    name: str = "chaios_knowledge.db"
    user: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: str = "disable"
    connection_pool_size: int = 10
    connection_timeout: int = 30

@dataclass
class CacheConfig:
    """Cache configuration"""
    type: str = "redis"  # redis, memcached, memory
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ttl: int = 3600  # Default TTL in seconds

@dataclass
class AIConfig:
    """AI/ML configuration"""
    cudnt_enabled: bool = True
    quantum_simulator_enabled: bool = True
    gpu_acceleration: bool = True
    model_cache_size: int = 100
    max_concurrent_models: int = 5
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    knowledge_expansion_rate: int = 1000  # Documents per hour

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    cors_origins: List[str] = None
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 100
    encryption_key: str = ""
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8000"]

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    enabled: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    alert_email: Optional[str] = None
    prometheus_enabled: bool = False
    grafana_enabled: bool = False

@dataclass
class PlatformConfig:
    """Main platform configuration"""
    environment: Environment = Environment.DEVELOPMENT
    version: str = "1.0.0"
    name: str = "chAIos Polymath Brain Platform"

    # Service configurations
    services: Dict[str, ServiceConfig] = None

    # Infrastructure
    database: DatabaseConfig = None
    cache: CacheConfig = None

    # Core systems
    ai: AIConfig = None
    security: SecurityConfig = None
    monitoring: MonitoringConfig = None

    # Feature flags
    features: Dict[str, bool] = None

    def __post_init__(self):
        if self.services is None:
            self.services = self._get_default_services()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.ai is None:
            self.ai = AIConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.features is None:
            self.features = self._get_default_features()

    def _get_default_services(self) -> Dict[str, ServiceConfig]:
        """Get default service configurations"""
        return {
            'main_api': ServiceConfig(
                name='Main API Server',
                port=8000,
                workers=4,
                dependencies=[]
            ),
            'enhanced_api': ServiceConfig(
                name='Enhanced API Server',
                port=8001,
                workers=2,
                dependencies=['main_api']
            ),
            'auth_service': ServiceConfig(
                name='Authentication Service',
                port=8002,
                workers=2,
                dependencies=['main_api']
            ),
            'knowledge_rag': ServiceConfig(
                name='Knowledge RAG System',
                workers=1,
                dependencies=['enhanced_api']
            ),
            'polymath_brain': ServiceConfig(
                name='Polymath Brain Trainer',
                workers=1,
                dependencies=['knowledge_rag']
            ),
            'cudnt_accelerator': ServiceConfig(
                name='CUDNT Universal Accelerator',
                workers=1,
                dependencies=[]
            ),
            'quantum_simulator': ServiceConfig(
                name='Quantum Simulator',
                workers=1,
                dependencies=[]
            )
        }

    def _get_default_features(self) -> Dict[str, bool]:
        """Get default feature flags"""
        return {
            'knowledge_expansion': True,
            'polymath_training': True,
            'cross_domain_mapping': True,
            'consciousness_enhancement': True,
            'quantum_acceleration': True,
            'gpu_acceleration': True,
            'real_time_learning': True,
            'api_documentation': True,
            'monitoring_dashboard': True,
            'security_hardening': True
        }

class ConfigurationManager:
    """Central configuration management system"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.current_config: Optional[PlatformConfig] = None
        self.config_cache = {}

        # Configuration file paths
        self.main_config_file = self.config_dir / "platform_config.json"
        self.env_config_file = self.config_dir / "environment_config.json"
        self.secrets_file = self.config_dir / "secrets.json"

        # Load initial configuration
        self.load_configuration()

    def load_configuration(self, environment: Optional[Environment] = None) -> PlatformConfig:
        """Load platform configuration"""
        if environment is None:
            environment = self._detect_environment()

        # Load base configuration
        base_config = self._load_base_config()

        # Load environment-specific configuration
        env_config = self._load_environment_config(environment)

        # Merge configurations
        merged_config = self._merge_configs(base_config, env_config)

        # Load secrets
        secrets = self._load_secrets()
        merged_config = self._apply_secrets(merged_config, secrets)

        # Override with environment variables
        merged_config = self._apply_environment_variables(merged_config)

        # Create PlatformConfig object
        self.current_config = PlatformConfig(**merged_config)

        logger.info(f"Configuration loaded for environment: {environment.value}")
        return self.current_config

    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env_var = os.getenv('ENVIRONMENT', os.getenv('ENV', 'development')).lower()

        env_mapping = {
            'dev': Environment.DEVELOPMENT,
            'development': Environment.DEVELOPMENT,
            'staging': Environment.STAGING,
            'stage': Environment.STAGING,
            'prod': Environment.PRODUCTION,
            'production': Environment.PRODUCTION,
            'test': Environment.TESTING,
            'testing': Environment.TESTING
        }

        return env_mapping.get(env_var, Environment.DEVELOPMENT)

    def _load_base_config(self) -> Dict[str, Any]:
        """Load base platform configuration"""
        if self.main_config_file.exists():
            with open(self.main_config_file, 'r') as f:
                return json.load(f)
        else:
            # Return default configuration
            return {
                'environment': 'development',
                'version': '1.0.0',
                'name': 'chAIos Polymath Brain Platform'
            }

    def _load_environment_config(self, environment: Environment) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        env_file = self.config_dir / f"{environment.value}_config.json"

        if env_file.exists():
            with open(env_file, 'r') as f:
                return json.load(f)

        # Return environment-specific defaults
        env_defaults = {
            Environment.DEVELOPMENT: {
                'debug': True,
                'log_level': 'DEBUG',
                'database': {'type': 'sqlite', 'name': 'chaios_knowledge_dev.db'},
                'cache': {'type': 'memory'},
                'monitoring': {'enabled': False}
            },
            Environment.STAGING: {
                'debug': False,
                'log_level': 'INFO',
                'database': {'type': 'postgresql', 'name': 'chaios_staging'},
                'cache': {'type': 'redis', 'host': 'redis-staging'},
                'monitoring': {'enabled': True, 'prometheus_enabled': True}
            },
            Environment.PRODUCTION: {
                'debug': False,
                'log_level': 'WARNING',
                'database': {'type': 'postgresql', 'name': 'chaios_production'},
                'cache': {'type': 'redis', 'host': 'redis-cluster'},
                'monitoring': {'enabled': True, 'prometheus_enabled': True, 'grafana_enabled': True},
                'security': {'ssl_enabled': True}
            },
            Environment.TESTING: {
                'debug': True,
                'log_level': 'DEBUG',
                'database': {'type': 'sqlite', 'name': ':memory:'},
                'cache': {'type': 'memory'}
            }
        }

        return env_defaults.get(environment, {})

    def _load_secrets(self) -> Dict[str, Any]:
        """Load secrets from secure storage"""
        if self.secrets_file.exists():
            with open(self.secrets_file, 'r') as f:
                return json.load(f)
        else:
            # Generate default secrets for development
            return {
                'jwt_secret_key': self._generate_secret_key(),
                'encryption_key': self._generate_secret_key(),
                'database_password': 'dev_password',
                'redis_password': ''
            }

    def _generate_secret_key(self, length: int = 32) -> str:
        """Generate a secure random key"""
        import secrets
        return secrets.token_hex(length)

    def _merge_configs(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries"""
        merged = base.copy()

        for key, value in overlay.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _apply_secrets(self, config: Dict[str, Any], secrets: Dict[str, Any]) -> Dict[str, Any]:
        """Apply secrets to configuration"""
        # Deep copy to avoid modifying original
        import copy
        config = copy.deepcopy(config)

        # Apply database secrets
        if 'database' in config:
            if 'password' in secrets:
                config['database']['password'] = secrets['password']

        # Apply security secrets
        if 'security' in config:
            if 'jwt_secret_key' in secrets:
                config['security']['jwt_secret_key'] = secrets['jwt_secret_key']
            if 'encryption_key' in secrets:
                config['security']['encryption_key'] = secrets['encryption_key']

        # Apply cache secrets
        if 'cache' in config:
            if 'password' in secrets.get('redis_password'):
                config['cache']['password'] = secrets['redis_password']

        return config

    def _apply_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        import copy
        config = copy.deepcopy(config)

        # Environment variable mappings
        env_mappings = {
            'DATABASE_URL': ('database', 'url'),
            'REDIS_URL': ('cache', 'url'),
            'JWT_SECRET_KEY': ('security', 'jwt_secret_key'),
            'ENCRYPTION_KEY': ('security', 'encryption_key'),
            'DEBUG': ('debug', lambda x: x.lower() == 'true'),
            'LOG_LEVEL': ('log_level',),
            'PORT': ('services', 'main_api', 'port', int)
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_config_value(config, config_path, env_value)

        return config

    def _set_nested_config_value(self, config: Dict[str, Any], path: tuple, value: Any):
        """Set a nested configuration value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        final_key = path[-1]
        if callable(final_key):
            current[path[-2]] = final_key(value)
        else:
            current[final_key] = value

    def save_configuration(self, config: PlatformConfig = None):
        """Save current configuration"""
        if config is None:
            config = self.current_config

        if config is None:
            logger.error("No configuration to save")
            return

        # Convert to dictionary
        config_dict = asdict(config)

        # Save main configuration
        with open(self.main_config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Configuration saved to {self.main_config_file}")

    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get configuration for a specific service"""
        if self.current_config and service_name in self.current_config.services:
            return self.current_config.services[service_name]
        return None

    def is_service_enabled(self, service_name: str) -> bool:
        """Check if a service is enabled"""
        service_config = self.get_service_config(service_name)
        return service_config.enabled if service_config else False

    def get_database_url(self) -> str:
        """Get database connection URL"""
        if not self.current_config:
            return "sqlite:///chaios_knowledge.db"

        db_config = self.current_config.database

        if db_config.type == "sqlite":
            return f"sqlite:///{db_config.name}"
        elif db_config.type == "postgresql":
            return f"postgresql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.name}"
        else:
            return f"{db_config.type}://{db_config.host}:{db_config.port}/{db_config.name}"

    def get_cache_url(self) -> str:
        """Get cache connection URL"""
        if not self.current_config:
            return "redis://localhost:6379"

        cache_config = self.current_config.cache

        if cache_config.password:
            return f"redis://:{cache_config.password}@{cache_config.host}:{cache_config.port}/{cache_config.db}"
        else:
            return f"redis://{cache_config.host}:{cache_config.port}/{cache_config.db}"

    def validate_configuration(self) -> List[str]:
        """Validate current configuration"""
        errors = []

        if not self.current_config:
            errors.append("No configuration loaded")
            return errors

        # Validate services
        for service_name, service_config in self.current_config.services.items():
            if service_config.port and (service_config.port < 1024 or service_config.port > 65535):
                errors.append(f"Invalid port {service_config.port} for service {service_name}")

            for dep in service_config.dependencies:
                if dep not in self.current_config.services:
                    errors.append(f"Service {service_name} depends on unknown service {dep}")

        # Validate database
        if self.current_config.database.type not in ['sqlite', 'postgresql', 'mongodb']:
            errors.append(f"Unsupported database type: {self.current_config.database.type}")

        # Validate cache
        if self.current_config.cache.type not in ['redis', 'memcached', 'memory']:
            errors.append(f"Unsupported cache type: {self.current_config.cache.type}")

        # Validate security
        if self.current_config.security.jwt_secret_key == "":
            errors.append("JWT secret key is not set")

        if self.current_config.security.encryption_key == "":
            errors.append("Encryption key is not set")

        return errors

    def create_environment_config(self, environment: Environment, config_overrides: Dict[str, Any] = None):
        """Create environment-specific configuration"""
        env_file = self.config_dir / f"{environment.value}_config.json"

        env_config = {
            'environment': environment.value,
            'debug': environment == Environment.DEVELOPMENT,
            'log_level': 'DEBUG' if environment == Environment.DEVELOPMENT else 'INFO'
        }

        if config_overrides:
            env_config.update(config_overrides)

        with open(env_file, 'w') as f:
            json.dump(env_config, f, indent=2)

        logger.info(f"Environment configuration created: {env_file}")

    def export_configuration(self, format: str = "json") -> str:
        """Export current configuration in specified format"""
        if not self.current_config:
            return ""

        config_dict = asdict(self.current_config)

        if format == "json":
            return json.dumps(config_dict, indent=2, default=str)
        elif format == "yaml":
            try:
                import yaml
                return yaml.dump(config_dict, default_flow_style=False)
            except ImportError:
                logger.warning("PyYAML not installed, falling back to JSON")
                return json.dumps(config_dict, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        if not self.current_config:
            return {'status': 'not_loaded'}

        enabled_services = [name for name, config in self.current_config.services.items() if config.enabled]
        enabled_features = [name for name, enabled in self.current_config.features.items() if enabled]

        return {
            'status': 'loaded',
            'environment': self.current_config.environment.value,
            'version': self.current_config.version,
            'services': {
                'total': len(self.current_config.services),
                'enabled': len(enabled_services),
                'disabled': len(self.current_config.services) - len(enabled_services)
            },
            'features': {
                'total': len(self.current_config.features),
                'enabled': len(enabled_features),
                'disabled': len(self.current_config.features) - len(enabled_features)
            },
            'infrastructure': {
                'database': self.current_config.database.type,
                'cache': self.current_config.cache.type,
                'monitoring': self.current_config.monitoring.enabled
            },
            'security': {
                'ssl_enabled': self.current_config.security.ssl_enabled,
                'rate_limiting': self.current_config.security.rate_limiting_enabled
            }
        }

def main():
    """Main function for configuration management"""
    import argparse

    parser = argparse.ArgumentParser(description='Configuration Management System')
    parser.add_argument('action', choices=['show', 'validate', 'save', 'export', 'summary', 'create-env'],
                       help='Action to perform')
    parser.add_argument('--env', choices=['development', 'staging', 'production', 'testing'],
                       help='Environment for configuration')
    parser.add_argument('--format', choices=['json', 'yaml'], default='json',
                       help='Export format')
    parser.add_argument('--output', help='Output file for export')

    args = parser.parse_args()

    manager = ConfigurationManager()

    try:
        if args.action == 'show':
            config = manager.current_config
            if config:
                print("Current Configuration:")
                print(manager.export_configuration(args.format))
            else:
                print("No configuration loaded")

        elif args.action == 'validate':
            errors = manager.validate_configuration()
            if errors:
                print("Configuration Validation Errors:")
                for error in errors:
                    print(f"  ❌ {error}")
                exit(1)
            else:
                print("✅ Configuration is valid")

        elif args.action == 'save':
            manager.save_configuration()
            print("✅ Configuration saved")

        elif args.action == 'export':
            config_str = manager.export_configuration(args.format)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(config_str)
                print(f"✅ Configuration exported to {args.output}")
            else:
                print(config_str)

        elif args.action == 'summary':
            summary = manager.get_configuration_summary()
            print("Configuration Summary:")
            print(json.dumps(summary, indent=2))

        elif args.action == 'create-env':
            if not args.env:
                print("Error: --env required for create-env action")
                exit(1)

            env_enum = Environment(args.env)
            manager.create_environment_config(env_enum)
            print(f"✅ Environment configuration created for {args.env}")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
