#!/usr/bin/env python3
"""
Comprehensive Test Suite for SquashPlot Core Components
======================================================

Tests for the core Chia farming management system including:
- Farming manager functionality
- Resource monitoring
- Plot analysis and optimization
- System integration tests

Author: Bradley Wallace (COO, Koba42 Corp)
"""

import os
import sys
import time
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import psutil
import json

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from squashplot_chia_system import (
    ChiaFarmingManager, OptimizationMode, PlotInfo,
    FarmingStats, SystemResources, SystemResourceMonitor,
    PlotOptimizer
)


class TestChiaFarmingManager(unittest.TestCase):
    """Test cases for the core Chia farming manager"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.plot_dirs = [os.path.join(self.temp_dir, f"plots{i}") for i in range(3)]

        # Create mock plot directories
        for plot_dir in self.plot_dirs:
            os.makedirs(plot_dir, exist_ok=True)

            # Create some mock plot files
            for i in range(5):
                plot_file = os.path.join(plot_dir, f"plot-k25-{i:04d}.plot")
                with open(plot_file, 'w') as f:
                    f.write("mock plot data")

        # Mock Chia root
        self.chia_root = os.path.join(self.temp_dir, "chia")

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test farming manager initialization"""
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs,
            optimization_mode=OptimizationMode.MIDDLE
        )

        self.assertEqual(manager.chia_root, self.chia_root)
        self.assertEqual(manager.plot_directories, self.plot_dirs)
        self.assertEqual(manager.optimization_mode, OptimizationMode.MIDDLE)
        self.assertIsInstance(manager.plots, list)
        self.assertIsInstance(manager.farming_stats, FarmingStats)

    def test_optimization_mode_speed(self):
        """Test speed optimization mode parameters"""
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs,
            optimization_mode=OptimizationMode.SPEED
        )

        self.assertEqual(manager.plot_threads, max(1, psutil.cpu_count() // 2))
        self.assertEqual(manager.farming_threads, max(2, psutil.cpu_count() - 2))
        self.assertEqual(manager.memory_buffer, 0.8)
        self.assertTrue(manager.gpu_acceleration)
        self.assertEqual(manager.plot_batch_size, 4)
        self.assertEqual(manager.farming_priority, "high")

    def test_optimization_mode_cost(self):
        """Test cost optimization mode parameters"""
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs,
            optimization_mode=OptimizationMode.COST
        )

        self.assertEqual(manager.plot_threads, 1)
        self.assertEqual(manager.farming_threads, 1)
        self.assertEqual(manager.memory_buffer, 0.3)
        self.assertFalse(manager.gpu_acceleration)
        self.assertEqual(manager.plot_batch_size, 1)
        self.assertEqual(manager.farming_priority, "low")

    def test_optimization_mode_middle(self):
        """Test middle (balanced) optimization mode parameters"""
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs,
            optimization_mode=OptimizationMode.MIDDLE
        )

        self.assertEqual(manager.plot_threads, max(2, psutil.cpu_count() // 3))
        self.assertEqual(manager.farming_threads, max(1, psutil.cpu_count() // 4))
        self.assertEqual(manager.memory_buffer, 0.5)
        self.assertEqual(manager.gpu_acceleration, True)  # Will be False if GPU not available
        self.assertEqual(manager.plot_batch_size, 2)
        self.assertEqual(manager.farming_priority, "normal")

    def test_scan_plot_directories(self):
        """Test plot directory scanning"""
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs
        )

        manager._scan_plot_directories()

        # Should find 15 plot files (5 per directory * 3 directories)
        self.assertEqual(len(manager.plots), 15)

        # Check plot information structure
        for plot in manager.plots:
            self.assertIsInstance(plot, PlotInfo)
            self.assertTrue(plot.filename.endswith('.plot'))
            self.assertIsInstance(plot.size_gb, float)
            self.assertIsInstance(plot.creation_time, datetime)
            self.assertIsInstance(plot.quality_score, float)
            self.assertIn(plot.location, self.plot_dirs)

    def test_analyze_plot_file(self):
        """Test individual plot file analysis"""
        manager = ChiaFarmingManager()

        # Create a mock plot file
        mock_plot = os.path.join(self.temp_dir, "test-plot-k25-0001.plot")
        with open(mock_plot, 'w') as f:
            f.write("x" * (100 * 1024 * 1024))  # 100MB mock file

        plot_info = manager._analyze_plot_file(mock_plot)

        self.assertIsNotNone(plot_info)
        self.assertEqual(plot_info.filename, "test-plot-k25-0001.plot")
        self.assertAlmostEqual(plot_info.size_gb, 0.1, places=1)
        self.assertIsInstance(plot_info.creation_time, datetime)
        self.assertGreater(plot_info.quality_score, 0)
        self.assertLess(plot_info.quality_score, 1)

    def test_update_farming_stats(self):
        """Test farming statistics update"""
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs
        )

        # First scan the directories to populate plots
        manager._scan_plot_directories()

        # Mock the Chia status method
        with patch.object(manager, '_get_chia_farming_status') as mock_status:
            mock_status.return_value = {
                'proofs_24h': 5,
                'network_space': 1000000.0,
                'balance': 100.5
            }

            manager._update_farming_stats()

            self.assertEqual(manager.farming_stats.total_plots, 15)
            self.assertEqual(manager.farming_stats.active_plots, 15)
            # Since we created 15 plots of ~0MB each (mock files), total size will be very small
            self.assertGreaterEqual(manager.farming_stats.total_size_gb, 0)
            self.assertEqual(manager.farming_stats.proofs_found_24h, 5)
            self.assertEqual(manager.farming_stats.network_space, 1000000.0)
            self.assertEqual(manager.farming_stats.farmer_balance, 100.5)

    def test_create_optimized_plot_plan(self):
        """Test optimized plot creation planning"""
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs,
            optimization_mode=OptimizationMode.SPEED
        )

        plan = manager.create_optimized_plot_plan(
            target_plots=10,
            available_space_gb=1000.0
        )

        self.assertIn('target_plots', plan)
        self.assertIn('plot_size_gb', plan)
        self.assertIn('total_space_required', plan)
        self.assertIn('optimization_mode', plan)
        self.assertIn('recommended_threads', plan)
        self.assertIn('gpu_accelerated', plan)
        self.assertIn('estimated_completion_hours', plan)

        self.assertEqual(plan['optimization_mode'], 'speed')
        self.assertLessEqual(plan['target_plots'], 10)

    def test_get_farming_report(self):
        """Test farming report generation"""
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs
        )

        report = manager.get_farming_report()

        required_keys = [
            'timestamp', 'farming_stats', 'system_resources',
            'optimization_mode', 'plot_details', 'recommendations'
        ]

        for key in required_keys:
            self.assertIn(key, report)

        self.assertIsInstance(report['timestamp'], str)
        self.assertIsInstance(report['recommendations'], list)


class TestSystemResourceMonitor(unittest.TestCase):
    """Test cases for system resource monitoring"""

    def setUp(self):
        self.monitor = SystemResourceMonitor()

    def test_get_resources(self):
        """Test resource data collection"""
        resources = self.monitor.get_resources()

        self.assertIsInstance(resources, SystemResources)
        self.assertIsInstance(resources.cpu_usage, float)
        self.assertIsInstance(resources.memory_usage, float)
        self.assertIsInstance(resources.disk_usage, dict)

        # CPU usage should be between 0 and 100
        self.assertGreaterEqual(resources.cpu_usage, 0)
        self.assertLessEqual(resources.cpu_usage, 100)

        # Memory usage should be between 0 and 100
        self.assertGreaterEqual(resources.memory_usage, 0)
        self.assertLessEqual(resources.memory_usage, 100)

    def test_network_io_monitoring(self):
        """Test network I/O monitoring"""
        # Get initial network stats
        initial_stats = self.monitor._get_network_io()

        # Wait a moment and get updated stats
        time.sleep(0.1)
        updated_stats = self.monitor._get_network_io()

        self.assertIsInstance(initial_stats, dict)
        self.assertIsInstance(updated_stats, dict)

        # Should have upload and download keys
        required_keys = ['upload_mbps', 'download_mbps']
        for key in required_keys:
            self.assertIn(key, initial_stats)
            self.assertIn(key, updated_stats)


class TestPlotOptimizer(unittest.TestCase):
    """Test cases for plot optimization"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.plot_dirs = [os.path.join(self.temp_dir, f"plots{i}") for i in range(3)]

        # Create mock farming manager
        self.farming_manager = Mock()
        self.farming_manager.plots = []
        self.farming_manager.optimization_mode = OptimizationMode.MIDDLE

        # Create some mock plots with proper datetime objects
        for i, plot_dir in enumerate(self.plot_dirs):
            for j in range(3):
                plot = Mock()
                plot.filename = f"plot-{i}-{j}.plot"
                plot.size_gb = 100.0 + j * 10
                plot.quality_score = 0.8 + j * 0.05
                plot.location = plot_dir
                # Use real datetime objects instead of Mock
                plot.creation_time = datetime.now() - timedelta(days=100 + j * 30)
                self.farming_manager.plots.append(plot)

        self.optimizer = PlotOptimizer(self.farming_manager)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_optimize_plot_distribution(self):
        """Test plot distribution optimization"""
        result = self.optimizer.optimize_plot_distribution()

        required_keys = [
            'current_distribution', 'recommendations',
            'f2_optimization'
        ]

        for key in required_keys:
            self.assertIn(key, result)

        self.assertIsInstance(result['current_distribution'], dict)
        self.assertIsInstance(result['recommendations'], list)
        self.assertIsInstance(result['f2_optimization'], dict)

    def test_f2_optimization_calculation(self):
        """Test F2 optimization algorithm"""
        # Create proper mock plots with datetime objects
        plot1 = Mock()
        plot1.quality_score = 0.8
        plot1.creation_time = datetime.now() - timedelta(days=100)

        plot2 = Mock()
        plot2.quality_score = 0.9
        plot2.creation_time = datetime.now() - timedelta(days=50)

        plot3 = Mock()
        plot3.quality_score = 0.7
        plot3.creation_time = datetime.now() - timedelta(days=200)

        plot4 = Mock()
        plot4.quality_score = 0.85
        plot4.creation_time = datetime.now() - timedelta(days=30)

        plot5 = Mock()
        plot5.quality_score = 0.75
        plot5.creation_time = datetime.now() - timedelta(days=150)

        plots_by_drive = {
            '/drive1': [plot1, plot2],
            '/drive2': [plot3, plot4],
            '/drive3': [plot5]
        }

        f2_metrics = self.optimizer._apply_f2_optimization(plots_by_drive)

        required_keys = [
            'plot_access_efficiency', 'drive_utilization_balance',
            'optimization_score'
        ]

        for key in required_keys:
            self.assertIn(key, f2_metrics)
            self.assertIsInstance(f2_metrics[key], float)
            self.assertGreaterEqual(f2_metrics[key], 0)
            self.assertLessEqual(f2_metrics[key], 1)

    def test_distribution_recommendations(self):
        """Test distribution recommendations generation"""
        # Create unbalanced distribution with proper datetime objects
        plots_drive1 = []
        plots_drive2 = []
        plots_drive3 = []

        for i in range(10):
            plot = Mock()
            plot.creation_time = datetime.now() - timedelta(days=100 + i)
            plots_drive1.append(plot)

        for i in range(2):
            plot = Mock()
            plot.creation_time = datetime.now() - timedelta(days=100 + i)
            plots_drive2.append(plot)

        for i in range(5):
            plot = Mock()
            plot.creation_time = datetime.now() - timedelta(days=100 + i)
            plots_drive3.append(plot)

        plots_by_drive = {
            '/drive1': plots_drive1,  # 10 plots
            '/drive2': plots_drive2,   # 2 plots
            '/drive3': plots_drive3    # 5 plots
        }

        recommendations = self.optimizer._generate_distribution_recommendations(plots_by_drive)

        self.assertIsInstance(recommendations, list)
        # Should recommend redistribution due to imbalance
        self.assertTrue(any("unbalanced" in rec.lower() for rec in recommendations))


class TestIntegration(unittest.TestCase):
    """Integration tests for SquashPlot components"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.chia_root = os.path.join(self.temp_dir, "chia")
        self.plot_dirs = [os.path.join(self.temp_dir, f"plots{i}") for i in range(2)]

        # Create directories and mock plots
        for plot_dir in self.plot_dirs:
            os.makedirs(plot_dir, exist_ok=True)
            for i in range(3):
                plot_file = os.path.join(plot_dir, f"plot-k25-{i:04d}.plot")
                with open(plot_file, 'w') as f:
                    f.write("mock")

    def tearDown(self):
        """Clean up integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_system_integration(self):
        """Test full system integration"""
        # Initialize farming manager
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs,
            optimization_mode=OptimizationMode.MIDDLE
        )

        # Start monitoring
        manager.start_monitoring()
        self.assertTrue(manager.monitoring_active)

        # Wait for monitoring cycle
        time.sleep(2)

        # Check that plots were scanned
        self.assertEqual(len(manager.plots), 6)  # 3 plots per directory * 2 directories

        # Get farming report
        report = manager.get_farming_report()
        self.assertIsNotNone(report)

        # Stop monitoring
        manager.stop_monitoring()
        self.assertFalse(manager.monitoring_active)

    def test_mode_switching_integration(self):
        """Test optimization mode switching"""
        manager = ChiaFarmingManager(
            chia_root=self.chia_root,
            plot_directories=self.plot_dirs
        )

        # Test all optimization modes
        for mode in [OptimizationMode.SPEED, OptimizationMode.COST, OptimizationMode.MIDDLE]:
            manager.optimization_mode = mode
            manager._set_optimization_parameters()

            # Verify parameters are set correctly
            self.assertIsNotNone(manager.plot_threads)
            self.assertIsNotNone(manager.farming_threads)
            self.assertIsNotNone(manager.memory_buffer)
            self.assertIsInstance(manager.gpu_acceleration, bool)
            self.assertIsNotNone(manager.plot_batch_size)
            self.assertIsNotNone(manager.farming_priority)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChiaFarmingManager)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSystemResourceMonitor))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlotOptimizer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print("SQUASHPLOT CORE TEST RESULTS")
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
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} TEST(S) FAILED!")

    print(f"{'='*60}")
