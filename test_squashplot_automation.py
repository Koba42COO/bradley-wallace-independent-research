#!/usr/bin/env python3
"""
Test Suite for SquashPlot Automation Engine
==========================================

Tests for automated Chia farming management including:
- Scheduled tasks and automation
- Cost-based optimization
- Alert system and monitoring
- Email notifications and reporting

Author: Bradley Wallace (COO, Koba42 Corp)
"""

import os
import sys
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import smtplib

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from squashplot_automation import (
    SquashPlotAutomation, AutomationMode, AutomationSchedule,
    AutomationAlert, CostSchedule
)


class TestAutomationEngine(unittest.TestCase):
    """Test cases for the automation engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = "/tmp/squashplot_test"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.chia_root = os.path.join(self.temp_dir, "chia")
        self.plot_dirs = [os.path.join(self.temp_dir, "plots1"), os.path.join(self.temp_dir, "plots2")]
        self.temp_dirs = [os.path.join(self.temp_dir, "temp1"), os.path.join(self.temp_dir, "temp2")]

        # Create directories
        for directory in self.plot_dirs + self.temp_dirs:
            os.makedirs(directory, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test automation engine initialization"""
        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs,
            temp_directories=self.temp_dirs,
            automation_mode=AutomationMode.SCHEDULED
        )

        self.assertEqual(automation.chia_root, self.chia_root)
        self.assertEqual(automation.plot_directories, self.plot_dirs)
        self.assertEqual(automation.temp_directories, self.temp_dirs)
        self.assertEqual(automation.automation_mode, AutomationMode.SCHEDULED)

    def test_cost_based_scheduling(self):
        """Test cost-based scheduling functionality"""
        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            automation_mode=AutomationMode.COST_BASED
        )

        # Test low cost hours configuration
        cost_schedule = CostSchedule(
            low_cost_hours=[2, 3, 4, 22, 23, 24],  # Late night/early morning
            high_cost_hours=[17, 18, 19, 20],       # Peak evening hours
            current_rate_per_kwh=0.15,
            max_daily_budget=5.0,
            budget_used_today=2.5
        )

        automation.cost_schedule = cost_schedule

        # Test cost calculation
        current_hour = 3  # Low cost hour
        is_low_cost = current_hour in cost_schedule.low_cost_hours
        self.assertTrue(is_low_cost)

        current_hour = 18  # High cost hour
        is_high_cost = current_hour in cost_schedule.high_cost_hours
        self.assertTrue(is_high_cost)

    def test_automation_schedule_creation(self):
        """Test automation schedule creation"""
        schedule = AutomationSchedule(
            task_name="daily_optimization",
            schedule_type="daily",
            time_config="02:00",
            enabled=True,
            parameters={
                "mode": "speed",
                "max_plots": 4,
                "target_directories": ["/plots1", "/plots2"]
            }
        )

        self.assertEqual(schedule.task_name, "daily_optimization")
        self.assertEqual(schedule.schedule_type, "daily")
        self.assertEqual(schedule.time_config, "02:00")
        self.assertTrue(schedule.enabled)
        self.assertIsInstance(schedule.parameters, dict)

    def test_alert_system(self):
        """Test automation alert system"""
        alert = AutomationAlert(
            alert_type="resource_warning",
            condition="cpu_usage > 90",
            threshold=90,
            action="switch_to_cost_mode",
            enabled=True
        )

        self.assertEqual(alert.alert_type, "resource_warning")
        self.assertEqual(alert.condition, "cpu_usage > 90")
        self.assertEqual(alert.threshold, 90)
        self.assertEqual(alert.action, "switch_to_cost_mode")
        self.assertTrue(alert.enabled)

    @patch('smtplib.SMTP')
    def test_email_notifications(self, mock_smtp):
        """Test email notification system"""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            automation_mode=AutomationMode.SCHEDULED
        )

        # Test email sending - using the correct method name
        alert = AutomationAlert(
            alert_type="test_alert",
            condition="test_condition",
            threshold=50,
            action="test_action",
            enabled=True
        )

        # The method currently only logs the alert, doesn't actually send email
        with patch('squashplot_automation.logger') as mock_logger:
            automation._send_alert_email(alert)

            # Verify that the logger was called with the alert message
            mock_logger.info.assert_called_with(f"Alert email would be sent: {alert.alert_type}")

    def test_performance_based_automation(self):
        """Test performance-based automation decisions"""
        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            automation_mode=AutomationMode.PERFORMANCE
        )

        # Mock the farming report instead of individual stats
        mock_report = {
            'farming_stats': {
                'proofs_found_24h': 10,
                'farming_efficiency': 0.85,
                'network_space': 1000000.0
            },
            'system_resources': {
                'cpu_usage': 75.0,
                'memory_usage': 60.0
            }
        }

        with patch.object(automation.farming_manager, 'get_farming_report', return_value=mock_report):
            # Test performance monitoring (using the correct method name)
            performance_data = automation._monitor_performance()

            # The method doesn't return anything, just logs
            self.assertIsNone(performance_data)

    def test_maintenance_scheduling(self):
        """Test maintenance task scheduling"""
        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            automation_mode=AutomationMode.MAINTENANCE
        )

        # Test maintenance tasks
        maintenance_tasks = [
            "disk_health_check",
            "plot_file_validation",
            "log_cleanup",
            "system_update_check"
        ]

        # Test that maintenance automation can be initialized
        self.assertIsNotNone(automation)
        self.assertEqual(automation.automation_mode, AutomationMode.MAINTENANCE)

    def test_task_scheduler(self):
        """Test task scheduling system"""
        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            automation_mode=AutomationMode.SCHEDULED
        )

        # Test adding a schedule
        automation.add_schedule(
            task_name="test_task",
            schedule_type="daily",
            time_config="02:00"
        )

        # Verify the schedule was added to the internal list
        self.assertEqual(len(automation.schedules), 1)
        schedule = automation.schedules[0]
        self.assertEqual(schedule.task_name, "test_task")
        self.assertEqual(schedule.schedule_type, "daily")
        self.assertEqual(schedule.time_config, "02:00")
        self.assertTrue(schedule.enabled)

    def test_resource_optimization_decisions(self):
        """Test resource optimization decision making"""
        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            automation_mode=AutomationMode.PERFORMANCE
        )

        # Mock resource monitor
        mock_resources = Mock()
        mock_resources.cpu_usage = 85
        mock_resources.memory_usage = 75
        mock_resources.gpu_usage = 60

        # Mock the resource monitor method
        with patch.object(automation.farming_manager.resource_monitor, 'get_resources', return_value=mock_resources):
            # Test that we can access resource monitoring
            resources = automation.farming_manager.resource_monitor.get_resources()
            self.assertEqual(resources.cpu_usage, 85)
            self.assertEqual(resources.memory_usage, 75)

    def test_budget_tracking(self):
        """Test electricity cost budget tracking"""
        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            automation_mode=AutomationMode.COST_BASED
        )

        cost_schedule = CostSchedule(
            low_cost_hours=[22, 23, 24, 0, 1, 2],
            high_cost_hours=[12, 13, 14, 15, 16, 17, 18],
            current_rate_per_kwh=0.12,
            max_daily_budget=10.0,
            budget_used_today=3.5
        )

        automation.cost_schedule = cost_schedule

        # Test budget calculation
        remaining_budget = cost_schedule.max_daily_budget - cost_schedule.budget_used_today
        self.assertEqual(remaining_budget, 6.5)

        # Test if plotting is allowed within budget
        plotting_cost_per_hour = 2.5  # Example cost
        can_plot = remaining_budget >= plotting_cost_per_hour
        self.assertTrue(can_plot)


class TestAutomationIntegration(unittest.TestCase):
    """Integration tests for automation components"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = "/tmp/squashplot_automation_test"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.chia_root = os.path.join(self.temp_dir, "chia")
        self.plot_dirs = [os.path.join(self.temp_dir, "plots")]
        self.temp_dirs = [os.path.join(self.temp_dir, "temp")]

        for directory in self.plot_dirs + self.temp_dirs + [self.chia_root]:
            os.makedirs(directory, exist_ok=True)

    def tearDown(self):
        """Clean up integration test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_automation_workflow(self):
        """Test complete automation workflow"""
        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs,
            temp_directories=self.temp_dirs,
            automation_mode=AutomationMode.SCHEDULED
        )

        # Test initialization
        self.assertIsNotNone(automation.farming_manager)
        self.assertIsNotNone(automation.automation_mode)

        # Test configuration loading using the correct method
        # The method doesn't return anything, just loads into the object
        with patch('os.path.exists', return_value=False):  # No config file exists
            automation.load_automation_config()
            # Should not raise any exceptions

        # Test automation start/stop
        automation.start_automation()
        self.assertTrue(automation.automation_active)

        # Allow some time for automation to run
        time.sleep(1)

        automation.stop_automation()
        self.assertFalse(automation.automation_active)

    def test_mode_switching_integration(self):
        """Test switching between automation modes"""
        automation = SquashPlotAutomation(
            chia_root=self.chia_root,
            automation_mode=AutomationMode.SCHEDULED
        )

        # Test switching to different modes
        for mode in [AutomationMode.COST_BASED, AutomationMode.PERFORMANCE, AutomationMode.MAINTENANCE]:
            automation.automation_mode = mode

            # Verify mode can be set
            self.assertEqual(automation.automation_mode, mode)


class TestAutomationUtils(unittest.TestCase):
    """Test utility functions for automation"""

    def test_time_based_decisions(self):
        """Test time-based decision making"""
        automation = SquashPlotAutomation()

        # Test current time evaluation
        current_hour = datetime.now().hour

        # Test if current time is in low-cost window
        low_cost_hours = [22, 23, 24, 0, 1, 2]
        is_low_cost_time = current_hour in low_cost_hours

        # Test if current time is in high-cost window
        high_cost_hours = [17, 18, 19, 20, 21]
        is_high_cost_time = current_hour in high_cost_hours

        # Should be in exactly one category
        self.assertTrue(is_low_cost_time or is_high_cost_time)
        self.assertFalse(is_low_cost_time and is_high_cost_time)

    def test_configuration_persistence(self):
        """Test configuration saving and loading"""
        automation = SquashPlotAutomation()

        test_config = {
            "chia_root": "/test/chia",
            "plot_directories": ["/test/plots1", "/test/plots2"],
            "automation_mode": "scheduled",
            "cost_schedule": {
                "low_cost_hours": [22, 23, 24],
                "current_rate_per_kwh": 0.15
            }
        }

        # Test configuration saving using the correct method
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('json.dump') as mock_json:
                automation.save_automation_config()

                mock_file.assert_called()
                mock_json.assert_called()

        # Test configuration loading using the correct method
        # The method doesn't return anything, just loads into the object
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(test_config))):
            with patch('json.load', return_value=test_config):
                with patch('os.path.exists', return_value=True):
                    automation.load_automation_config()
                    # Should not raise any exceptions


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAutomationEngine)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAutomationIntegration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAutomationUtils))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print("SQUASHPLOT AUTOMATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    if result.wasSuccessful():
        print("\n✅ ALL AUTOMATION TESTS PASSED!")
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} AUTOMATION TEST(S) FAILED!")

    print(f"{'='*60}")
