#!/usr/bin/env python3
"""
Final Product Launcher - Unified Orchestration System
====================================================
Complete launcher for the chAIos Polymath Brain Platform
Integrates all tools, systems, and services into a unified product.
"""

import os
import sys
import json
import time
import subprocess
import threading
import signal
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_product_launcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FinalProductLauncher:
    """Unified launcher for the complete chAIos platform"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config = self._load_config()
        self.processes = {}
        self.services = {}
        self.is_running = False

        # Define all available services
        self.available_services = {
            # Core AI/ML Systems
            'cudnt_accelerator': {
                'name': 'CUDNT Universal Accelerator',
                'script': 'cudnt_universal_accelerator.py',
                'port': None,
                'description': 'High-performance GPU acceleration system'
            },
            'quantum_simulator': {
                'name': 'Quantum Annealing Simulator',
                'script': 'gpu_quantum_accelerator.py',
                'port': None,
                'description': 'Quantum computing simulation'
            },

            # Knowledge Systems
            'knowledge_rag': {
                'name': 'Advanced Agentic RAG System',
                'script': 'advanced_agentic_rag_system.py',
                'port': None,
                'description': 'Intelligent document retrieval and synthesis'
            },
            'polymath_brain': {
                'name': 'Polymath Brain Trainer',
                'script': 'polymath_brain_trainer.py',
                'port': None,
                'description': 'Continuous knowledge expansion system'
            },

            # API Services
            'main_api': {
                'name': 'Main API Server',
                'script': 'api_server.py',
                'port': 8000,
                'description': 'Primary REST API service'
            },
            'enhanced_api': {
                'name': 'Enhanced API Server',
                'script': 'enhanced_api_server.py',
                'port': 8001,
                'description': 'Advanced API with prime aligned compute features'
            },

            # Web Scraping Systems
            'knowledge_scraper': {
                'name': 'Knowledge Web Scraper',
                'script': 'web_scraper_knowledge_system.py',
                'port': None,
                'description': 'Automated knowledge acquisition'
            },

            # Educational Systems
            'learning_pathways': {
                'name': 'Learning Pathway System',
                'script': 'learning_pathway_system.py',
                'port': None,
                'description': 'Personalized learning journeys'
            },

            # Authentication & Security
            'auth_service': {
                'name': 'Authentication Service',
                'script': 'auth_service.py',
                'port': 8002,
                'description': 'User authentication and authorization'
            },

            # Utility Services
            'redis_cache': {
                'name': 'Redis Cache Setup',
                'script': 'redis_cache_setup.py',
                'port': 6379,
                'description': 'High-performance caching system'
            }
        }

        # Service dependencies
        self.service_dependencies = {
            'main_api': [],
            'enhanced_api': ['main_api'],
            'auth_service': ['main_api'],
            'knowledge_rag': ['enhanced_api'],
            'polymath_brain': ['knowledge_rag'],
            'knowledge_scraper': ['knowledge_rag'],
            'learning_pathways': ['polymath_brain']
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config files"""
        config = {
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'auto_start': os.getenv('AUTO_START_SERVICES', '').split(',') if os.getenv('AUTO_START_SERVICES') else [],
            'ports': {
                'main_api': int(os.getenv('MAIN_API_PORT', '8000')),
                'enhanced_api': int(os.getenv('ENHANCED_API_PORT', '8001')),
                'auth_service': int(os.getenv('AUTH_SERVICE_PORT', '8002')),
                'redis': int(os.getenv('REDIS_PORT', '6379'))
            },
            'databases': {
                'main': os.getenv('DATABASE_URL', 'sqlite:///chaios_knowledge.db'),
                'cache': os.getenv('REDIS_URL', 'redis://localhost:6379')
            }
        }

        # Load from config file if exists
        config_file = self.base_dir / 'product_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)

        return config

    def save_config(self):
        """Save current configuration"""
        config_file = self.base_dir / 'product_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {config_file}")

    def list_services(self) -> List[Dict[str, Any]]:
        """List all available services"""
        services = []
        for service_id, service_info in self.available_services.items():
            status = 'running' if service_id in self.processes else 'stopped'
            services.append({
                'id': service_id,
                'name': service_info['name'],
                'status': status,
                'port': service_info.get('port'),
                'description': service_info['description'],
                'dependencies': self.service_dependencies.get(service_id, [])
            })
        return services

    def check_service_dependencies(self, service_id: str) -> bool:
        """Check if service dependencies are met"""
        dependencies = self.service_dependencies.get(service_id, [])
        for dep in dependencies:
            if dep not in self.processes:
                logger.warning(f"Service {service_id} requires {dep} to be running")
                return False
        return True

    def start_service(self, service_id: str) -> bool:
        """Start a specific service"""
        if service_id not in self.available_services:
            logger.error(f"Unknown service: {service_id}")
            return False

        if service_id in self.processes:
            logger.info(f"Service {service_id} is already running")
            return True

        service_info = self.available_services[service_id]

        # Check dependencies
        if not self.check_service_dependencies(service_id):
            return False

        # Check if script exists
        script_path = self.base_dir / service_info['script']
        if not script_path.exists():
            logger.error(f"Service script not found: {script_path}")
            return False

        try:
            # Set environment variables for the service
            env = os.environ.copy()
            if service_info.get('port'):
                env['PORT'] = str(service_info['port'])

            # Start the service
            logger.info(f"Starting service: {service_info['name']}")
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_dir
            )

            self.processes[service_id] = process
            self.services[service_id] = service_info

            # Wait a moment for service to start
            time.sleep(2)

            # Check if process is still running
            if process.poll() is None:
                logger.info(f"✅ Service {service_id} started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ Service {service_id} failed to start")
                if stderr:
                    logger.error(f"Error: {stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"Failed to start service {service_id}: {e}")
            return False

    def stop_service(self, service_id: str) -> bool:
        """Stop a specific service"""
        if service_id not in self.processes:
            logger.info(f"Service {service_id} is not running")
            return True

        try:
            process = self.processes[service_id]
            process.terminate()

            # Wait for process to terminate
            try:
                process.wait(timeout=10)
                logger.info(f"✅ Service {service_id} stopped successfully")
            except subprocess.TimeoutExpired:
                logger.warning(f"Service {service_id} did not terminate gracefully, force killing")
                process.kill()
                process.wait()

            del self.processes[service_id]
            return True

        except Exception as e:
            logger.error(f"Failed to stop service {service_id}: {e}")
            return False

    def start_all_services(self, service_list: Optional[List[str]] = None) -> bool:
        """Start all services in dependency order"""
        if service_list is None:
            service_list = list(self.available_services.keys())

        # Resolve dependencies and start services in order
        started_services = set()
        max_attempts = len(service_list) * 2  # Prevent infinite loops

        for attempt in range(max_attempts):
            progress_made = False

            for service_id in service_list:
                if service_id in started_services:
                    continue

                # Check if all dependencies are met
                dependencies = self.service_dependencies.get(service_id, [])
                if all(dep in started_services for dep in dependencies):
                    if self.start_service(service_id):
                        started_services.add(service_id)
                        progress_made = True

            if not progress_made:
                break

        if len(started_services) == len(service_list):
            logger.info("✅ All requested services started successfully")
            return True
        else:
            failed_services = set(service_list) - started_services
            logger.error(f"❌ Failed to start services: {failed_services}")
            return False

    def stop_all_services(self):
        """Stop all running services"""
        logger.info("Stopping all services...")

        for service_id in list(self.processes.keys()):
            self.stop_service(service_id)

        logger.info("✅ All services stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'overall_status': 'running' if self.is_running else 'stopped',
            'services': {},
            'system_info': {
                'total_services': len(self.available_services),
                'running_services': len(self.processes),
                'config_environment': self.config['environment']
            },
            'health_checks': {}
        }

        # Service status
        for service_id, service_info in self.available_services.items():
            is_running = service_id in self.processes
            status['services'][service_id] = {
                'name': service_info['name'],
                'status': 'running' if is_running else 'stopped',
                'port': service_info.get('port'),
                'description': service_info['description']
            }

            # Health check for running services
            if is_running and service_info.get('port'):
                status['health_checks'][service_id] = self._check_service_health(service_id)

        return status

    def _check_service_health(self, service_id: str) -> Dict[str, Any]:
        """Check health of a running service"""
        service_info = self.available_services[service_id]
        port = service_info.get('port')

        if not port:
            return {'status': 'unknown', 'message': 'No health check available'}

        try:
            import requests
            # Try different health endpoints
            health_urls = [
                f"http://localhost:{port}/health",
                f"http://localhost:{port}/status",
                f"http://localhost:{port}/"
            ]

            for url in health_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        return {'status': 'healthy', 'url': url}
                except:
                    continue

            return {'status': 'unreachable', 'message': 'Service not responding'}

        except ImportError:
            return {'status': 'unknown', 'message': 'Requests library not available'}

    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        diagnostics = {
            'timestamp': time.time(),
            'system_check': {},
            'service_check': {},
            'performance_metrics': {},
            'recommendations': []
        }

        # System checks
        diagnostics['system_check'] = {
            'python_version': sys.version,
            'working_directory': str(self.base_dir),
            'environment': self.config['environment'],
            'debug_mode': self.config['debug']
        }

        # Service checks
        for service_id, service_info in self.available_services.items():
            script_path = self.base_dir / service_info['script']
            diagnostics['service_check'][service_id] = {
                'script_exists': script_path.exists(),
                'script_path': str(script_path),
                'has_dependencies': self.check_service_dependencies(service_id),
                'is_running': service_id in self.processes
            }

        # Performance metrics
        diagnostics['performance_metrics'] = {
            'total_services': len(self.available_services),
            'running_services': len(self.processes),
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage()
        }

        # Generate recommendations
        diagnostics['recommendations'] = self._generate_recommendations(diagnostics)

        return diagnostics

    def _get_memory_usage(self) -> str:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return ".1f"
        except ImportError:
            return "psutil not available"

    def _get_cpu_usage(self) -> str:
        """Get current CPU usage"""
        try:
            import psutil
            return ".1f"
        except ImportError:
            return "psutil not available"

    def _generate_recommendations(self, diagnostics: Dict[str, Any]) -> List[str]:
        """Generate system recommendations"""
        recommendations = []

        # Check for missing scripts
        missing_scripts = []
        for service_id, checks in diagnostics['service_check'].items():
            if not checks['script_exists']:
                missing_scripts.append(service_id)

        if missing_scripts:
            recommendations.append(f"Missing service scripts: {', '.join(missing_scripts)}")

        # Check service dependencies
        unmet_deps = []
        for service_id, checks in diagnostics['service_check'].items():
            if not checks['has_dependencies']:
                unmet_deps.append(service_id)

        if unmet_deps:
            recommendations.append(f"Services with unmet dependencies: {', '.join(unmet_deps)}")

        # Performance recommendations
        running_count = diagnostics['performance_metrics']['running_services']
        if running_count == 0:
            recommendations.append("No services are currently running. Consider starting core services.")
        elif running_count < len(self.available_services) * 0.5:
            recommendations.append("Only a few services are running. Consider starting more services for full functionality.")

        return recommendations

    def create_deployment_package(self, output_dir: str = "deployment_package"):
        """Create a deployment package with all necessary files"""
        output_path = self.base_dir / output_dir
        output_path.mkdir(exist_ok=True)

        logger.info(f"Creating deployment package at {output_path}")

        # Core system files
        core_files = [
            'final_product_launcher.py',
            'api_server.py',
            'enhanced_api_server.py',
            'auth_service.py',
            'database_service.py',
            'start_system.sh',
            'requirements.txt',
            'pyproject.toml'
        ]

        # Knowledge systems
        knowledge_files = [
            'knowledge_system_integration.py',
            'advanced_agentic_rag_system.py',
            'polymath_brain_trainer.py',
            'massive_knowledge_expansion.py',
            'cross_domain_mapper.py'
        ]

        # AI/ML systems
        ai_files = [
            'cudnt_universal_accelerator.py',
            'gpu_quantum_accelerator.py',
            'wallace_math_engine.py'
        ]

        # Educational systems
        education_files = [
            'learning_pathway_system.py',
            'consciousness_enhanced_learning.py'
        ]

        # Configuration files
        config_files = [
            'product_config.json',
            'build_config.json'
        ]

        # Docker files
        docker_files = [
            'Dockerfile',
            'docker-compose.yml'
        ]

        all_files = core_files + knowledge_files + ai_files + education_files + config_files + docker_files

        # Copy files
        for file_name in all_files:
            src_path = self.base_dir / file_name
            if src_path.exists():
                import shutil
                shutil.copy2(src_path, output_path)
                logger.info(f"Copied {file_name}")

        # Create deployment manifest
        manifest = {
            'name': 'chAIos Polymath Brain Platform',
            'version': '1.0.0',
            'description': 'Complete AI platform with knowledge systems, brain training, and polymath capabilities',
            'services': list(self.available_services.keys()),
            'main_launcher': 'final_product_launcher.py',
            'created_at': time.time(),
            'files': all_files
        }

        with open(output_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info("✅ Deployment package created successfully")
        return output_path

def main():
    """Main entry point for the final product launcher"""
    parser = argparse.ArgumentParser(description='chAIos Polymath Brain Platform Launcher')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'list', 'diagnostics', 'deploy'],
                       help='Action to perform')
    parser.add_argument('--services', nargs='+', help='Specific services to start/stop')
    parser.add_argument('--output-dir', default='deployment_package', help='Output directory for deployment')

    args = parser.parse_args()

    launcher = FinalProductLauncher()

    try:
        if args.action == 'list':
            services = launcher.list_services()
            print("
📋 Available Services:"            print("-" * 60)
            for service in services:
                status_icon = "🟢" if service['status'] == 'running' else "🔴"
                port_info = f" (Port: {service['port']})" if service['port'] else ""
                print(f"{status_icon} {service['id']}: {service['name']}{port_info}")
                print(f"   📝 {service['description']}")
                if service['dependencies']:
                    print(f"   🔗 Dependencies: {', '.join(service['dependencies'])}")
                print()

        elif args.action == 'start':
            if args.services:
                success = launcher.start_all_services(args.services)
            else:
                # Auto-start core services
                core_services = ['main_api', 'enhanced_api', 'auth_service', 'knowledge_rag']
                success = launcher.start_all_services(core_services)

            if success:
                print("✅ Services started successfully")
                print("\nPress Ctrl+C to stop all services")

                # Keep running and monitor
                launcher.is_running = True
                try:
                    while launcher.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n⏹️ Shutting down services...")
                    launcher.stop_all_services()
            else:
                print("❌ Failed to start services")
                sys.exit(1)

        elif args.action == 'stop':
            launcher.stop_all_services()
            print("✅ All services stopped")

        elif args.action == 'status':
            status = launcher.get_system_status()
            print("
📊 System Status:"            print("-" * 40)
            print(f"Overall Status: {'🟢 RUNNING' if status['overall_status'] == 'running' else '🔴 STOPPED'}")
            print(f"Total Services: {status['system_info']['total_services']}")
            print(f"Running Services: {status['system_info']['running_services']}")

            print("
🟢 Running Services:"            for service_id, service_info in status['services'].items():
                if service_info['status'] == 'running':
                    port_info = f" (Port: {service_info['port']})" if service_info['port'] else ""
                    health = status['health_checks'].get(service_id, {})
                    health_icon = "💚" if health.get('status') == 'healthy' else "🤍"
                    print(f"   {health_icon} {service_info['name']}{port_info}")

        elif args.action == 'diagnostics':
            diagnostics = launcher.run_diagnostics()
            print("
🔍 System Diagnostics:"            print("-" * 40)
            print(f"Python Version: {diagnostics['system_check']['python_version']}")
            print(f"Environment: {diagnostics['system_check']['environment']}")
            print(f"Total Services: {diagnostics['performance_metrics']['total_services']}")
            print(f"Running Services: {diagnostics['performance_metrics']['running_services']}")

            if diagnostics['recommendations']:
                print("
💡 Recommendations:"                for rec in diagnostics['recommendations']:
                    print(f"   • {rec}")

        elif args.action == 'deploy':
            output_path = launcher.create_deployment_package(args.output_dir)
            print(f"✅ Deployment package created at: {output_path}")

    except Exception as e:
        logger.error(f"Launcher error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
