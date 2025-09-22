#!/usr/bin/env python3
"""
SquashPlot Automation Engine - Automated Chia Farming Management
=================================================================

Intelligent automation system for Chia blockchain farming operations.
Features automated plotting schedules, maintenance tasks, and optimization routines.

Features:
- Automated plotting schedules based on electricity costs
- Predictive maintenance and disk health monitoring
- Smart resource allocation and optimization
- Cost-based automation decisions
- Emergency response and alert handling
- Performance trend analysis and adaptation

Author: Bradley Wallace (COO, Koba42 Corp)
Contact: user@domain.com
License: MIT License
"""

import os
import sys
import time
import json
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import SquashPlot components
from squashplot_chia_system import ChiaFarmingManager, OptimizationMode
from f2_gpu_optimizer import F2GPUOptimizer, PerformanceProfile
from squashplot_disk_optimizer import DiskOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('squashplot_automation')

class AutomationMode(Enum):
    """Automation operation modes"""
    SCHEDULED = "scheduled"
    COST_BASED = "cost_based"
    PERFORMANCE = "performance"
    MAINTENANCE = "maintenance"

@dataclass
class AutomationSchedule:
    """Automation schedule configuration"""
    task_name: str
    schedule_type: str  # daily, hourly, weekly
    time_config: str    # "14:30", "hourly", "monday 09:00"
    enabled: bool
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    parameters: Dict[str, Any] = None

@dataclass
class AutomationAlert:
    """Automation alert configuration"""
    alert_type: str
    condition: str
    threshold: Any
    action: str
    enabled: bool
    last_triggered: Optional[datetime] = None

@dataclass
class CostSchedule:
    """Electricity cost-based scheduling"""
    low_cost_hours: List[int]  # Hours with low electricity cost
    high_cost_hours: List[int]  # Hours with high electricity cost
    current_rate_per_kwh: float
    max_daily_budget: float
    budget_used_today: float

class SquashPlotAutomation:
    """Automated Chia farming management system"""

    def __init__(self, chia_root: str = "~/chia-blockchain",
                 plot_directories: List[str] = None,
                 temp_directories: List[str] = None,
                 automation_mode: AutomationMode = AutomationMode.SCHEDULED):
        """
        Initialize automation engine

        Args:
            chia_root: Path to Chia blockchain installation
            plot_directories: Plot storage directories
            temp_directories: Temporary plotting directories
            automation_mode: Primary automation strategy
        """
        self.chia_root = os.path.expanduser(chia_root)
        self.plot_directories = plot_directories or []
        self.temp_directories = temp_directories or []
        self.automation_mode = automation_mode

        # Initialize components
        self.farming_manager = ChiaFarmingManager(
            chia_root=chia_root,
            plot_directories=plot_directories,
            optimization_mode=OptimizationMode.MIDDLE
        )

        self.gpu_optimizer = F2GPUOptimizer(
            chia_root=chia_root,
            temp_dirs=temp_directories or [],
            final_dirs=plot_directories or [],
            profile=PerformanceProfile.MIDDLE
        )

        self.disk_optimizer = DiskOptimizer(
            plot_directories=plot_directories or []
        )

        # Automation state
        self.schedules: List[AutomationSchedule] = []
        self.alerts: List[AutomationAlert] = []
        self.cost_schedule = CostSchedule(
            low_cost_hours=[22, 23, 0, 1, 2, 3, 4, 5],  # 10 PM - 6 AM
            high_cost_hours=[12, 13, 14, 15, 16, 17, 18],  # 12 PM - 7 PM
            current_rate_per_kwh=0.12,
            max_daily_budget=5.0,
            budget_used_today=0.0
        )

        # Control flags
        self.automation_active = False
        self.monitoring_active = False

        # Load configuration
        self.load_automation_config()

        logger.info(f"SquashPlot Automation initialized with {automation_mode.value} mode")

    def start_automation(self):
        """Start the automation engine"""
        if self.automation_active:
            logger.warning("Automation already active")
            return

        self.automation_active = True
        self.monitoring_active = True

        # Start farming monitoring
        self.farming_manager.start_monitoring()

        # Start automation threads
        automation_thread = threading.Thread(target=self._automation_loop, daemon=True)
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)

        automation_thread.start()
        monitoring_thread.start()

        # Setup scheduled tasks
        self._setup_scheduled_tasks()

        logger.info("SquashPlot Automation started")

    def stop_automation(self):
        """Stop the automation engine"""
        self.automation_active = False
        self.monitoring_active = False

        self.farming_manager.stop_monitoring()

        logger.info("SquashPlot Automation stopped")

    def _automation_loop(self):
        """Main automation loop"""
        while self.automation_active:
            try:
                # Run automation tasks based on mode
                if self.automation_mode == AutomationMode.COST_BASED:
                    self._run_cost_based_automation()
                elif self.automation_mode == AutomationMode.PERFORMANCE:
                    self._run_performance_based_automation()
                elif self.automation_mode == AutomationMode.MAINTENANCE:
                    self._run_maintenance_automation()
                else:  # SCHEDULED
                    self._run_scheduled_automation()

                time.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Automation loop error: {e}")
                time.sleep(60)

    def _monitoring_loop(self):
        """Continuous monitoring and alert checking"""
        while self.monitoring_active:
            try:
                # Check alerts
                self._check_automation_alerts()

                # Update cost tracking
                self._update_cost_tracking()

                # Performance monitoring
                self._monitor_performance()

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)

    def _setup_scheduled_tasks(self):
        """Setup scheduled automation tasks"""
        # Daily optimization at 2 AM
        schedule.every().day.at("02:00").do(self._daily_optimization)

        # Hourly health check
        schedule.every().hour.do(self._hourly_health_check)

        # Weekly disk optimization (Sundays at 3 AM)
        schedule.every().sunday.at("03:00").do(self._weekly_disk_optimization)

        # Cost-based plotting (check every hour)
        schedule.every().hour.do(self._cost_based_plotting_check)

        logger.info("Scheduled tasks configured")

    def _run_cost_based_automation(self):
        """Run cost-based automation decisions"""
        current_hour = datetime.now().hour
        current_cost = self._get_current_electricity_cost()

        # Plot during low-cost hours
        if current_hour in self.cost_schedule.low_cost_hours:
            if self.cost_schedule.budget_used_today < self.cost_schedule.max_daily_budget:
                self._initiate_cost_optimized_plotting()

        # Optimize during high-cost hours
        elif current_hour in self.cost_schedule.high_cost_hours:
            self.farming_manager.optimization_mode = OptimizationMode.COST
            self.farming_manager._set_optimization_parameters()

    def _run_performance_based_automation(self):
        """Run performance-based automation"""
        # Analyze current performance
        farming_report = self.farming_manager.get_farming_report()

        cpu_usage = farming_report['system_resources']['cpu_usage']
        memory_usage = farming_report['system_resources']['memory_usage']

        # Adjust based on resource usage
        if cpu_usage > 90 or memory_usage > 85:
            # Switch to cost mode to reduce resource usage
            self.farming_manager.optimization_mode = OptimizationMode.COST
            logger.info("Switched to COST mode due to high resource usage")
        elif cpu_usage < 50 and memory_usage < 60:
            # Switch to speed mode for better performance
            self.farming_manager.optimization_mode = OptimizationMode.SPEED
            logger.info("Switched to SPEED mode for better performance")
        else:
            # Balanced mode
            self.farming_manager.optimization_mode = OptimizationMode.MIDDLE

        self.farming_manager._set_optimization_parameters()

    def _run_maintenance_automation(self):
        """Run maintenance-based automation"""
        # Check disk health
        health_report = self.disk_optimizer.get_disk_health_report()

        if health_report['health_status'] in ['poor', 'critical']:
            logger.warning("Disk health issues detected - running optimization")
            self.disk_optimizer.optimize_disk_space()

        # Check farming efficiency
        farming_stats = self.farming_manager.farming_stats
        if farming_stats.proofs_found_24h == 0 and farming_stats.total_plots > 0:
            logger.warning("No proofs found - checking farming configuration")
            self._check_farming_configuration()

    def _run_scheduled_automation(self):
        """Run scheduled automation tasks"""
        schedule.run_pending()

    def _daily_optimization(self):
        """Daily optimization routine"""
        logger.info("Running daily optimization")

        # Full system optimization
        self.farming_manager.optimization_mode = OptimizationMode.MIDDLE
        self.farming_manager._set_optimization_parameters()

        # Disk optimization
        self.disk_optimizer.optimize_disk_space()

        # Reset daily budget
        self.cost_schedule.budget_used_today = 0.0

        logger.info("Daily optimization completed")

    def _hourly_health_check(self):
        """Hourly health check"""
        logger.info("Running hourly health check")

        # System health
        farming_report = self.farming_manager.get_farming_report()

        # Resource check
        resources = farming_report['system_resources']
        if resources['cpu_usage'] > 95:
            logger.critical("CRITICAL: CPU usage above 95%")
        elif resources['memory_usage'] > 90:
            logger.critical("CRITICAL: Memory usage above 90%")

    def _weekly_disk_optimization(self):
        """Weekly disk optimization"""
        logger.info("Running weekly disk optimization")

        # Deep disk analysis and optimization
        optimization = self.disk_optimizer.optimize_disk_space()

        # Log optimization results
        if optimization['optimization_summary']['status'] == 'optimization_required':
            logger.info(f"Disk optimization completed: {optimization['optimization_summary']}")

    def _cost_based_plotting_check(self):
        """Check if cost-based plotting should be initiated"""
        current_hour = datetime.now().hour

        if current_hour in self.cost_schedule.low_cost_hours:
            remaining_budget = self.cost_schedule.max_daily_budget - self.cost_schedule.budget_used_today

            if remaining_budget > 0.5:  # At least $0.50 remaining
                self._initiate_cost_optimized_plotting()

    def _initiate_cost_optimized_plotting(self):
        """Initiate cost-optimized plotting"""
        logger.info("Initiating cost-optimized plotting")

        # Check available resources
        system_analysis = self.gpu_optimizer._analyze_system_state()

        if system_analysis['total_memory_gb'] > 16:  # Sufficient memory
            # Start 1-2 plots based on available resources
            num_plots = min(2, int(system_analysis['total_memory_gb'] / 8))

            # Use GPU if available during low-cost hours
            profile = PerformanceProfile.SPEED if system_analysis['gpu_available'] else PerformanceProfile.COST

            # This would integrate with actual Chia keys
            logger.info(f"Would start {num_plots} plots with {profile.value} profile")

    def _check_automation_alerts(self):
        """Check and trigger automation alerts"""
        for alert in self.alerts:
            if not alert.enabled:
                continue

            if self._check_alert_condition(alert):
                self._trigger_alert(alert)

    def _check_alert_condition(self, alert: AutomationAlert) -> bool:
        """Check if alert condition is met"""
        if alert.alert_type == "cpu_usage":
            resources = self.farming_manager.resource_monitor.get_resources()
            return resources['cpu_usage'] > alert.threshold

        elif alert.alert_type == "disk_space":
            for disk in self.disk_optimizer.disk_info.values():
                if disk.usage_percent > alert.threshold:
                    return True

        elif alert.alert_type == "plotting_failure":
            # Check recent plotting failures
            return False  # Placeholder

        return False

    def _trigger_alert(self, alert: AutomationAlert):
        """Trigger automation alert"""
        logger.warning(f"Alert triggered: {alert.alert_type} - {alert.condition}")

        alert.last_triggered = datetime.now()

        # Execute alert action
        if alert.action == "email":
            self._send_alert_email(alert)
        elif alert.action == "optimize":
            self.farming_manager.optimization_mode = OptimizationMode.COST
        elif alert.action == "restart":
            logger.info("Alert action: restart requested")

    def _send_alert_email(self, alert: AutomationAlert):
        """Send alert email notification"""
        try:
            # Email configuration (would be loaded from config)
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "user@domain.com"
            receiver_email = "user@domain.com"

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = f"SquashPlot Alert: {alert.alert_type}"

            body = f"""
            SquashPlot Automation Alert

            Type: {alert.alert_type}
            Condition: {alert.condition}
            Threshold: {alert.threshold}
            Time: {datetime.now().isoformat()}

            Farming Status: {self.farming_manager.farming_stats.total_plots} plots
            """

            msg.attach(MIMEText(body, 'plain'))

            # This would actually send the email
            logger.info(f"Alert email would be sent: {alert.alert_type}")

        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")

    def _update_cost_tracking(self):
        """Update electricity cost tracking"""
        # This would integrate with electricity monitoring APIs
        # For now, use simple estimation
        pass

    def _monitor_performance(self):
        """Monitor system performance trends"""
        # Track performance metrics over time
        farming_report = self.farming_manager.get_farming_report()

        # Log significant changes
        cpu_usage = farming_report['system_resources']['cpu_usage']
        if cpu_usage > 85:
            logger.info(f"High CPU usage detected: {cpu_usage}%")

    def _get_current_electricity_cost(self) -> float:
        """Get current electricity cost"""
        current_hour = datetime.now().hour

        if current_hour in self.cost_schedule.low_cost_hours:
            return self.cost_schedule.current_rate_per_kwh * 0.7  # 30% discount
        elif current_hour in self.cost_schedule.high_cost_hours:
            return self.cost_schedule.current_rate_per_kwh * 1.5  # 50% premium
        else:
            return self.cost_schedule.current_rate_per_kwh

    def _check_farming_configuration(self):
        """Check and validate farming configuration"""
        logger.info("Checking farming configuration")

        # Verify plot files exist
        total_plots = 0
        for plot_dir in self.plot_directories:
            if os.path.exists(plot_dir):
                plot_files = [f for f in os.listdir(plot_dir) if f.endswith('.plot')]
                total_plots += len(plot_files)

        if total_plots == 0:
            logger.error("No plot files found - farming configuration issue")
        else:
            logger.info(f"Farming configuration OK: {total_plots} plots found")

    def add_schedule(self, task_name: str, schedule_type: str,
                    time_config: str, parameters: Dict[str, Any] = None):
        """Add automation schedule"""
        schedule = AutomationSchedule(
            task_name=task_name,
            schedule_type=schedule_type,
            time_config=time_config,
            enabled=True,
            last_run=None,
            next_run=None,
            parameters=parameters or {}
        )

        self.schedules.append(schedule)
        logger.info(f"Added automation schedule: {task_name}")

    def add_alert(self, alert_type: str, condition: str,
                 threshold: Any, action: str):
        """Add automation alert"""
        alert = AutomationAlert(
            alert_type=alert_type,
            condition=condition,
            threshold=threshold,
            action=action,
            enabled=True,
            last_triggered=None
        )

        self.alerts.append(alert)
        logger.info(f"Added automation alert: {alert_type}")

    def get_automation_status(self) -> Dict[str, Any]:
        """Get automation system status"""
        return {
            'automation_active': self.automation_active,
            'monitoring_active': self.monitoring_active,
            'mode': self.automation_mode.value,
            'schedules': len(self.schedules),
            'alerts': len(self.alerts),
            'farming_stats': asdict(self.farming_manager.farming_stats),
            'cost_budget': {
                'used_today': self.cost_schedule.budget_used_today,
                'max_daily': self.cost_schedule.max_daily_budget,
                'remaining': self.cost_schedule.max_daily_budget - self.cost_schedule.budget_used_today
            }
        }

    def load_automation_config(self, config_file: str = "squashplot_automation.json"):
        """Load automation configuration"""
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Load schedules
                for schedule_data in config.get('schedules', []):
                    schedule = AutomationSchedule(**schedule_data)
                    self.schedules.append(schedule)

                # Load alerts
                for alert_data in config.get('alerts', []):
                    alert = AutomationAlert(**alert_data)
                    self.alerts.append(alert)

                # Load cost schedule
                cost_data = config.get('cost_schedule', {})
                for key, value in cost_data.items():
                    if hasattr(self.cost_schedule, key):
                        setattr(self.cost_schedule, key, value)

                logger.info(f"Loaded automation config from {config_file}")

            except Exception as e:
                logger.error(f"Failed to load automation config: {e}")

    def save_automation_config(self, config_file: str = "squashplot_automation.json"):
        """Save automation configuration"""
        config = {
            'schedules': [asdict(s) for s in self.schedules],
            'alerts': [asdict(a) for a in self.alerts],
            'cost_schedule': asdict(self.cost_schedule)
        }

        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.info(f"Saved automation config to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save automation config: {e}")

def main():
    """Main automation application"""
    import argparse

    parser = argparse.ArgumentParser(description='SquashPlot Automation - Automated Chia Farming')
    parser.add_argument('--chia-root', default='~/chia-blockchain',
                       help='Path to Chia blockchain installation')
    parser.add_argument('--plot-dirs', nargs='+', required=True,
                       help='Plot directories to monitor')
    parser.add_argument('--temp-dirs', nargs='+',
                       help='Temporary plotting directories')
    parser.add_argument('--mode', choices=['scheduled', 'cost_based', 'performance', 'maintenance'],
                       default='scheduled', help='Automation mode')
    parser.add_argument('--config', help='Automation configuration file')
    parser.add_argument('--setup', action='store_true',
                       help='Setup automation with default schedules')

    args = parser.parse_args()

    # Initialize automation
    mode = AutomationMode(args.mode)
    automation = SquashPlotAutomation(
        chia_root=args.chia_root,
        plot_directories=args.plot_dirs,
        temp_directories=args.temp_dirs,
        automation_mode=mode
    )

    if args.config:
        automation.load_automation_config(args.config)

    if args.setup:
        # Setup default automation schedules
        automation.add_schedule(
            "daily_optimization",
            "daily",
            "02:00",
            {"task": "full_system_optimization"}
        )

        automation.add_schedule(
            "hourly_health_check",
            "hourly",
            "hourly",
            {"task": "system_health_check"}
        )

        # Setup default alerts
        automation.add_alert(
            "cpu_usage",
            "CPU usage above 90%",
            90,
            "email"
        )

        automation.add_alert(
            "disk_space",
            "Disk usage above 95%",
            95,
            "optimize"
        )

        # Save configuration
        automation.save_automation_config()

    try:
        print("🤖 SquashPlot Automation Engine")
        print("=" * 40)
        print(f"Mode: {mode.value.upper()}")
        print(f"Monitoring {len(args.plot_dirs)} plot directories")
        print("Press Ctrl+C to stop")
        print("=" * 40)

        # Start automation
        automation.start_automation()

        # Main loop
        while True:
            status = automation.get_automation_status()
            print(f"📊 Status: Mode={status['mode']}, "
                  f"Plots={status['farming_stats']['total_plots']}, "
                  f"Budget=${status['cost_budget']['remaining']:.2f} remaining")
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n🛑 Stopping SquashPlot Automation...")
    finally:
        automation.stop_automation()

if __name__ == '__main__':
    main()
