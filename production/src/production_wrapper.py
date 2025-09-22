#!/usr/bin/env python3
"""
Production wrapper for SquashPlot Enhanced
Provides production-ready entry point with enhanced logging and monitoring
"""

import sys
import os
import logging
import signal
import time
import json
from pathlib import Path

# Production configuration constants
REQUIRED_DISK_SPACE_GB = 250  # Minimum space for plotting
REQUIRED_MEMORY_GB = 8       # Minimum memory
MAX_CPU_PERCENT = 95         # CPU threshold
MAX_MEMORY_PERCENT = 90      # Memory threshold

try:
    from squashplot_enhanced import SquashPlotEnhanced, PlotConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def setup_production_logging():
    """Configure logging for production environment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/tmp/squashplot_production.log'),
            logging.StreamHandler()
        ]
    )

def get_production_config():
    """Get production configuration from environment"""
    return {
        'production_mode': os.getenv('PRODUCTION_MODE', 'true').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'default_plot_dir': os.getenv('DEFAULT_PLOT_DIR', '/plots'),
        'default_temp_dir': os.getenv('DEFAULT_TEMP_DIR', '/tmp/squashplot'),
        'max_concurrent_plots': int(os.getenv('MAX_CONCURRENT_PLOTS', '4')),
        'timeout_plotting': int(os.getenv('TIMEOUT_PLOTTING', '7200')),
        'timeout_compression': int(os.getenv('TIMEOUT_COMPRESSION', '3600')),
        'monitor_interval': int(os.getenv('MONITOR_INTERVAL', '30')),
        'required_disk_space_gb': REQUIRED_DISK_SPACE_GB,
        'required_memory_gb': REQUIRED_MEMORY_GB
    }

class ProductionSquashPlot:
    """Production wrapper for SquashPlot Enhanced"""
    
    def __init__(self):
        # Setup production logging
        setup_production_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load production configuration
        self.config = get_production_config()
        
        # Initialize SquashPlot Enhanced
        self.squashplot = SquashPlotEnhanced()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info("SquashPlot Enhanced Production Wrapper initialized")
        
    def start_web_server(self):
        """Start the web server in production mode"""
        self.logger.info("Starting SquashPlot web server in production mode")
        
        # Set production environment variables
        os.environ['FLASK_ENV'] = 'production'
        os.environ['FLASK_DEBUG'] = 'false'
        
        # Import and start the web server
        try:
            from web_server import app
            app.run(
                host='0.0.0.0',
                port=5000,
                debug=False,
                use_reloader=False
            )
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            return False
        
        return True
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
        
    def health_check(self):
        """Perform authoritative system health check for production"""
        self.logger.info("Performing production health check...")
        
        health_status = {
            'tools_available': False,
            'disk_space_sufficient': False,
            'memory_sufficient': False,
            'cpu_usage_normal': False,
            'overall_healthy': False
        }
        
        try:
            import psutil
            
            # Check tool availability (CRITICAL - fail if missing)
            tool_validation = self.squashplot.tool_manager.validate_tools()
            has_plotting_tools = (tool_validation['madmax_available'] or 
                                tool_validation['bladebit_available'])
            
            if not has_plotting_tools:
                self.logger.error("CRITICAL: No plotting tools available (Mad Max or BladeBit required)")
                health_status['tools_available'] = False
            else:
                self.logger.info("✅ Plotting tools available")
                health_status['tools_available'] = True
            
            # Check disk space in temp directory (CRITICAL)
            temp_dir = self.config['default_temp_dir']
            try:
                os.makedirs(temp_dir, exist_ok=True)
                usage = psutil.disk_usage(temp_dir)
                free_gb = usage.free / (1024**3)
                required_gb = self.config['required_disk_space_gb']
                
                if free_gb < required_gb:
                    self.logger.error(f"CRITICAL: Insufficient disk space: {free_gb:.1f}GB available, {required_gb}GB required")
                    health_status['disk_space_sufficient'] = False
                else:
                    self.logger.info(f"✅ Disk space sufficient: {free_gb:.1f}GB available")
                    health_status['disk_space_sufficient'] = True
            except Exception as e:
                self.logger.error(f"CRITICAL: Cannot access temp directory {temp_dir}: {e}")
                health_status['disk_space_sufficient'] = False
                
            # Check memory (CRITICAL)
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            required_gb = self.config['required_memory_gb']
            
            if total_gb < required_gb:
                self.logger.error(f"CRITICAL: Insufficient memory: {total_gb:.1f}GB total, {required_gb}GB required")
                health_status['memory_sufficient'] = False
            elif memory.percent > MAX_MEMORY_PERCENT:
                self.logger.error(f"CRITICAL: High memory usage: {memory.percent:.1f}% (max {MAX_MEMORY_PERCENT}%)")
                health_status['memory_sufficient'] = False
            else:
                self.logger.info(f"✅ Memory sufficient: {total_gb:.1f}GB total, {memory.percent:.1f}% used")
                health_status['memory_sufficient'] = True
                
            # Check CPU usage (WARNING only)
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > MAX_CPU_PERCENT:
                self.logger.warning(f"High CPU usage: {cpu_percent:.1f}% (max {MAX_CPU_PERCENT}%)")
                health_status['cpu_usage_normal'] = False
            else:
                self.logger.info(f"✅ CPU usage normal: {cpu_percent:.1f}%")
                health_status['cpu_usage_normal'] = True
                
            # Overall health determination
            critical_checks = [
                health_status['tools_available'],
                health_status['disk_space_sufficient'], 
                health_status['memory_sufficient']
            ]
            
            health_status['overall_healthy'] = all(critical_checks)
            
            if health_status['overall_healthy']:
                self.logger.info("✅ Production health check PASSED")
            else:
                self.logger.error("❌ Production health check FAILED")
                
            # Log structured health status
            self.logger.info(f"Health status: {json.dumps(health_status, indent=2)}")
            
            return health_status['overall_healthy']
            
        except Exception as e:
            self.logger.error(f"Health check exception: {e}")
            health_status['overall_healthy'] = False
            return False
            
    def plot_with_monitoring(self, plot_config: PlotConfig):
        """Execute plotting with production monitoring"""
        self.logger.info("Starting production plotting with enhanced monitoring")
        
        start_time = time.time()
        
        try:
            # Pre-flight checks
            if not self.health_check():
                raise RuntimeError("Health check failed")
                
            # Execute plotting
            result = self.squashplot.plot(plot_config)
            
            duration = time.time() - start_time
            
            if result.success:
                self.logger.info(f"Plotting completed successfully in {duration:.1f}s")
                self.logger.info(f"Plot size: {result.size_gb:.1f}GB")
                if hasattr(result, 'compression_ratio'):
                    self.logger.info(f"Compression ratio: {result.compression_ratio:.2f}")
            else:
                self.logger.error(f"Plotting failed: {result.error_message}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Production plotting failed: {e}")
            raise
            
    def run_service(self):
        """Run as a service (for deployment)"""
        self.logger.info("Starting SquashPlot Enhanced service mode")
        
        while True:
            try:
                # Perform periodic health checks
                self.health_check()
                time.sleep(self.config['monitor_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("Service stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Service error: {e}")
                time.sleep(60)  # Wait before retry

def main():
    """Production entry point with proper exit codes"""
    if len(sys.argv) < 2:
        print("SquashPlot Enhanced Production Wrapper")
        print("Usage: python production_wrapper.py [health-check|service|--help]")
        print("Commands:")
        print("  health-check  - Perform production readiness check")
        print("  service       - Run as persistent service")
        print("  --help        - Show tool status and compression levels")
        sys.exit(1)
        
    command = sys.argv[1]
    
    try:
        wrapper = ProductionSquashPlot()
        
        if command == "health-check":
            success = wrapper.health_check()
            if success:
                print("✅ Health check passed")
                sys.exit(0)
            else:
                print("❌ Health check failed")
                sys.exit(1)
                
        elif command == "service":
            wrapper.run_service()
            
        elif command == "--help":
            wrapper.squashplot.check_tools()
            wrapper.squashplot.list_compression_levels()
            sys.exit(0)
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'health-check', 'service', or '--help'")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Production wrapper failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()