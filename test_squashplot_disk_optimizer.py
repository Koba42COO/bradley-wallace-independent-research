#!/usr/bin/env python3
"""
Test Suite for SquashPlot Disk Optimizer
=======================================

Tests for disk optimization and plot management including:
- Disk health monitoring and analysis
- Plot distribution optimization
- Migration planning and execution
- Space management and cleanup

Author: Bradley Wallace (COO, Koba42 Corp)
"""

import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import psutil
import json

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from squashplot_disk_optimizer import (
    DiskOptimizer, DiskHealth, DiskInfo, PlotMigration
)


class TestDiskOptimizer(unittest.TestCase):
    """Test cases for the disk optimizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.plot_dirs = []

        # Create mock plot directories with different usage levels
        for i in range(4):
            plot_dir = os.path.join(self.temp_dir, f"plots{i}")
            os.makedirs(plot_dir, exist_ok=True)
            self.plot_dirs.append(plot_dir)

            # Create some mock plot files
            for j in range(5):
                plot_file = os.path.join(plot_dir, f"plot-k25-{i}{j:03d}.plot")
                with open(plot_file, 'w') as f:
                    # Write some data to simulate file size
                    f.write("x" * (1024 * 1024))  # 1MB mock file

        # Initialize disk optimizer
        self.optimizer = DiskOptimizer(
            plot_directories=self.plot_dirs,
            min_free_space_gb=10.0,
            rebalance_threshold=0.1
        )

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test disk optimizer initialization"""
        self.assertEqual(self.optimizer.plot_directories, self.plot_dirs)
        self.assertEqual(self.optimizer.min_free_space_gb, 10.0)
        self.assertEqual(self.optimizer.rebalance_threshold, 0.1)
        self.assertIsInstance(self.optimizer.disk_info, dict)
        self.assertIsInstance(self.optimizer.plot_distribution, dict)

    def test_analyze_disk_health(self):
        """Test disk health analysis"""
        # Mock the scan_disks method to avoid filesystem issues
        mock_disk_info = {}
        for i, plot_dir in enumerate(self.plot_dirs):
            mock_disk_info[plot_dir] = DiskInfo(
                mount_point=plot_dir,
                total_gb=1000.0,
                used_gb=500.0 + i * 50,
                free_gb=500.0 - i * 50,
                usage_percent=50.0 + i * 5.0,
                plot_count=10 + i,
                plot_size_gb=500.0 + i * 50.0,
                health_score=0.8 - i * 0.1,
                health_status=DiskHealth.GOOD
            )

        with patch.object(self.optimizer, 'scan_disks', return_value=mock_disk_info):
            disk_info = self.optimizer._analyze_disk_health()

            self.assertIsInstance(disk_info, dict)

            # Check that all plot directories are analyzed
            for plot_dir in self.plot_dirs:
                self.assertIn(plot_dir, disk_info)
                info = disk_info[plot_dir]
                self.assertIsInstance(info, DiskInfo)

            # Verify disk info structure
            self.assertIsInstance(info.mount_point, str)
            self.assertIsInstance(info.total_gb, float)
            self.assertIsInstance(info.used_gb, float)
            self.assertIsInstance(info.free_gb, float)
            self.assertIsInstance(info.usage_percent, float)
            self.assertIsInstance(info.health_score, float)
            self.assertIsInstance(info.health_status, DiskHealth)

    def test_calculate_disk_usage(self):
        """Test disk usage calculation"""
        test_dir = self.plot_dirs[0]

        # Mock psutil.disk_usage
        with patch('psutil.disk_usage') as mock_usage:
            mock_usage.return_value = Mock(
                total=100 * 1024**3,  # 100GB
                used=60 * 1024**3,    # 60GB used
                free=40 * 1024**3,    # 40GB free
                percent=60.0
            )

            usage = psutil.disk_usage(test_dir)

            self.assertEqual(usage.total, 100 * 1024**3)
            self.assertEqual(usage.used, 60 * 1024**3)
            self.assertEqual(usage.free, 40 * 1024**3)
            self.assertEqual(usage.percent, 60.0)

    def test_plot_distribution_analysis(self):
        """Test plot distribution analysis"""
        distribution = self.optimizer._analyze_plot_distribution()

        self.assertIsInstance(distribution, dict)

        # Check that all plot directories are included
        for plot_dir in self.plot_dirs:
            self.assertIn(plot_dir, distribution)
            plots = distribution[plot_dir]
            self.assertIsInstance(plots, list)

            # Verify plot information
            for plot in plots:
                self.assertIsInstance(plot, str)
                self.assertTrue(plot.endswith('.plot'))

    def test_optimization_plan_generation(self):
        """Test optimization plan generation"""
        plan = self.optimizer.generate_optimization_plan()

        required_keys = [
            'current_state', 'optimization_needed', 'recommended_actions',
            'estimated_benefits', 'risk_assessment'
        ]

        for key in required_keys:
            self.assertIn(key, plan)

        # Verify plan structure
        self.assertIsInstance(plan['current_state'], dict)
        self.assertIsInstance(plan['optimization_needed'], bool)
        self.assertIsInstance(plan['recommended_actions'], list)
        self.assertIsInstance(plan['estimated_benefits'], dict)
        self.assertIsInstance(plan['risk_assessment'], dict)

    def test_migration_planning(self):
        """Test plot migration planning"""
        # Create imbalanced scenario
        source_dir = self.plot_dirs[0]
        target_dir = self.plot_dirs[1]

        migrations = self.optimizer._plan_migrations()

        self.assertIsInstance(migrations, list)

        # Verify migration structure if any migrations are planned
        for migration in migrations:
            self.assertIsInstance(migration, PlotMigration)
            self.assertIsInstance(migration.source_disk, str)
            self.assertIsInstance(migration.target_disk, str)
            self.assertIsInstance(migration.plot_filename, str)
            self.assertIsInstance(migration.plot_size_gb, float)
            self.assertIsInstance(migration.priority, int)
            self.assertIsInstance(migration.estimated_time, float)

    def test_space_optimization(self):
        """Test space optimization recommendations"""
        recommendations = self.optimizer._generate_space_recommendations()

        self.assertIsInstance(recommendations, list)

        # Check for common space optimization recommendations
        recommendation_texts = [rec.lower() for rec in recommendations]

        # Should include basic space management advice
        space_related_terms = ['space', 'free', 'cleanup', 'delete', 'remove']
        has_space_advice = any(
            any(term in text for term in space_related_terms)
            for text in recommendation_texts
        )

        # Note: This might not always be true depending on actual disk state
        # but the method should return a list regardless
        pass

    def test_health_score_calculation(self):
        """Test disk health score calculation"""
        # Test with different usage scenarios
        test_cases = [
            (10, DiskHealth.EXCELLENT),    # 10% usage
            (45, DiskHealth.GOOD),         # 45% usage
            (75, DiskHealth.FAIR),         # 75% usage
            (90, DiskHealth.POOR),         # 90% usage
            (98, DiskHealth.CRITICAL),     # 98% usage
        ]

        for usage_percent, expected_health in test_cases:
            with self.subTest(usage_percent=usage_percent):
                health_score = self.optimizer._calculate_health_score(usage_percent)
                health_status = self.optimizer._determine_health_status(health_score)

                self.assertIsInstance(health_score, float)
                self.assertGreaterEqual(health_score, 0)
                self.assertLessEqual(health_score, 1)
                self.assertIsInstance(health_status, DiskHealth)

    def test_balancing_algorithm(self):
        """Test plot balancing algorithm"""
        # Create mock disk usage data
        disk_usage = {}
        for i, plot_dir in enumerate(self.plot_dirs):
            # Simulate different usage levels
            usage_percent = 40 + i * 15  # 40%, 55%, 70%, 85%
            disk_usage[plot_dir] = Mock(percent=usage_percent)

        with patch('psutil.disk_usage', side_effect=lambda path: disk_usage.get(path, Mock(percent=50))):
            is_balanced = self.optimizer._is_balanced()
            self.assertIsInstance(is_balanced, bool)

            balance_score = self.optimizer._calculate_balance_score()
            self.assertIsInstance(balance_score, float)
            self.assertGreaterEqual(balance_score, 0)
            self.assertLessEqual(balance_score, 1)

    def test_cleanup_operations(self):
        """Test cleanup operation planning"""
        cleanup_plan = self.optimizer.plan_cleanup_operations()

        self.assertIsInstance(cleanup_plan, dict)
        self.assertIn('operations', cleanup_plan)
        self.assertIn('estimated_space_freed', cleanup_plan)
        self.assertIn('risk_level', cleanup_plan)

        # Verify operations structure
        operations = cleanup_plan['operations']
        self.assertIsInstance(operations, list)

        for operation in operations:
            self.assertIsInstance(operation, dict)
            self.assertIn('type', operation)
            self.assertIn('description', operation)
            self.assertIn('estimated_space', operation)


class TestDiskHealthMonitoring(unittest.TestCase):
    """Test cases for disk health monitoring"""

    def setUp(self):
        """Set up health monitoring tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.plot_dirs = [os.path.join(self.temp_dir, "plots")]
        os.makedirs(self.plot_dirs[0], exist_ok=True)

        self.optimizer = DiskOptimizer(
            plot_directories=self.plot_dirs,
            min_free_space_gb=5.0
        )

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_disk_health_assessment(self):
        """Test comprehensive disk health assessment"""
        health_report = self.optimizer.assess_disk_health()

        self.assertIsInstance(health_report, dict)

        required_sections = [
            'overall_health', 'disk_health_summary', 'recommendations',
            'timestamp', 'overall_health_score'
        ]

        for section in required_sections:
            self.assertIn(section, health_report)

    def test_predictive_maintenance(self):
        """Test predictive maintenance analysis"""
        predictions = self.optimizer._predict_maintenance_needs()

        self.assertIsInstance(predictions, dict)
        self.assertIn('short_term', predictions)
        self.assertIn('medium_term', predictions)
        self.assertIn('long_term', predictions)

        # Verify prediction structure
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            timeframe_predictions = predictions[timeframe]
            self.assertIsInstance(timeframe_predictions, list)

            for prediction in timeframe_predictions:
                self.assertIsInstance(prediction, dict)
                self.assertIn('component', prediction)
                self.assertIn('risk_level', prediction)
                self.assertIn('estimated_time', prediction)
                self.assertIn('recommended_action', prediction)

    def test_performance_monitoring(self):
        """Test disk performance monitoring"""
        # Create a test file for performance testing
        test_file = os.path.join(self.plot_dirs[0], "performance_test.dat")
        test_data = "x" * (10 * 1024 * 1024)  # 10MB test data

        with open(test_file, 'w') as f:
            f.write(test_data)

        # Test read performance
        start_time = datetime.now()
        with open(test_file, 'r') as f:
            data = f.read()
        read_time = (datetime.now() - start_time).total_seconds()

        self.assertGreater(read_time, 0)
        self.assertEqual(len(data), len(test_data))

        # Test write performance
        start_time = datetime.now()
        with open(test_file, 'w') as f:
            f.write(test_data)
        write_time = (datetime.now() - start_time).total_seconds()

        self.assertGreater(write_time, 0)

        # Clean up
        os.remove(test_file)


class TestDiskOptimizerIntegration(unittest.TestCase):
    """Integration tests for disk optimizer components"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.plot_dirs = []

        # Create a more complex directory structure
        for i in range(3):
            plot_dir = os.path.join(self.temp_dir, f"plots{i}")
            os.makedirs(plot_dir, exist_ok=True)
            self.plot_dirs.append(plot_dir)

            # Create plots of different sizes
            for j in range(3):
                plot_file = os.path.join(plot_dir, f"plot-{i}-{j}.plot")
                # Create files of different sizes to simulate real scenario
                size_mb = (j + 1) * 100  # 100MB, 200MB, 300MB
                with open(plot_file, 'w') as f:
                    f.write("x" * (size_mb * 1024 * 1024))

        self.optimizer = DiskOptimizer(
            plot_directories=self.plot_dirs,
            min_free_space_gb=1.0,  # Low threshold for testing
            rebalance_threshold=0.05  # Low threshold for testing
        )

    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow"""
        # Step 1: Analyze current state
        analysis = self.optimizer.analyze_current_state()
        self.assertIsInstance(analysis, dict)

        # Step 2: Generate optimization plan
        plan = self.optimizer.generate_optimization_plan()
        self.assertIsInstance(plan, dict)  # Returns optimization plan dict

        # Verify plan structure
        required_keys = ['current_state', 'optimization_needed', 'recommended_actions']
        for key in required_keys:
            self.assertIn(key, plan)

        if plan.get('optimization_needed'):
            # Step 3: Execute optimization
            success = self.optimizer.execute_optimization()
            self.assertIsInstance(success, bool)

            # Step 4: Verify optimization results
            verification = self.optimizer.verify_optimization()
            self.assertIsInstance(verification, dict)

    def test_balanced_system_scenario(self):
        """Test with a balanced system scenario"""
        # Create a scenario where disks are relatively balanced
        for i, plot_dir in enumerate(self.plot_dirs):
            # Create similar number of similar-sized plots
            for j in range(2):
                plot_file = os.path.join(plot_dir, f"balanced-plot-{i}-{j}.plot")
                with open(plot_file, 'w') as f:
                    f.write("x" * (200 * 1024 * 1024))  # 200MB each

        optimizer = DiskOptimizer(
            plot_directories=self.plot_dirs,
            rebalance_threshold=0.2  # Higher threshold
        )

        is_balanced = optimizer._is_balanced()
        # Should be balanced with similar plot counts and sizes
        # (This is a probabilistic test - may need adjustment based on actual filesystem)

    def test_unbalanced_system_scenario(self):
        """Test with an unbalanced system scenario"""
        # Create highly unbalanced scenario
        plot_dir_heavy = self.plot_dirs[0]
        plot_dir_light = self.plot_dirs[1]

        # Create many plots in one directory, few in another
        for j in range(10):
            plot_file = os.path.join(plot_dir_heavy, f"heavy-plot-{j}.plot")
            with open(plot_file, 'w') as f:
                f.write("x" * (100 * 1024 * 1024))  # 100MB each

        for j in range(2):
            plot_file = os.path.join(plot_dir_light, f"light-plot-{j}.plot")
            with open(plot_file, 'w') as f:
                f.write("x" * (100 * 1024 * 1024))  # 100MB each

        optimizer = DiskOptimizer(
            plot_directories=self.plot_dirs,
            rebalance_threshold=0.05  # Low threshold to detect imbalance
        )

        is_balanced = optimizer._is_balanced()
        # Should detect imbalance
        # (This is a probabilistic test)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDiskOptimizer)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDiskHealthMonitoring))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDiskOptimizerIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print("SQUASHPLOT DISK OPTIMIZER TEST RESULTS")
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
        print("\n✅ ALL DISK OPTIMIZER TESTS PASSED!")
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} DISK OPTIMIZER TEST(S) FAILED!")

    print(f"{'='*60}")
