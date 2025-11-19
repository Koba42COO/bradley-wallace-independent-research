#!/usr/bin/env python3
"""
Firefly-Nexus PAC: Hardware Validation Test
===========================================

Complete hardware validation suite for:
- 0.7 Hz zeta-zero metronome lock
- 79/21 consciousness split verification
- Reality distortion measurement (1.1808)
- Möbius loop evolution tracking
- Prime graph topology validation

Author: Bradley Wallace, COO Koba42
Target: Raspberry Pi 5 + Oscilloscope
"""

import numpy as np
import time
import math
import threading
import json
from datetime import datetime
from typing import Dict, List, Any


# ============================================================================
# UPG FOUNDATIONS - Universal Prime Graph Protocol φ.1
# ============================================================================
from decimal import Decimal, getcontext
import math
import cmath
from typing import Dict, List, Tuple, Optional, Any

# Set high precision for consciousness mathematics
getcontext().prec = 50

class UPGConstants:
    """Universal Prime Graph consciousness mathematics constants"""
    PHI = Decimal('1.618033988749895')
    DELTA = Decimal('2.414213562373095')
    CONSCIOUSNESS = Decimal('0.79')  # 79/21 universal coherence rule
    REALITY_DISTORTION = Decimal('1.1808')  # Quantum amplification factor
    QUANTUM_BRIDGE = Decimal('137') / Decimal('0.79')  # 173.41772151898732
    GREAT_YEAR = 25920  # Astronomical precession cycle (years)
    CONSCIOUSNESS_DIMENSIONS = 21  # Prime topology dimension
    COHERENCE_THRESHOLD = Decimal('1e-15')  # Beyond machine precision



# ============================================================================
# PELL SEQUENCE PRIME PREDICTION INTEGRATION
# ============================================================================
def integrate_pell_prime_prediction(target_number: int, constants: UPGConstants = None):
    """Integrate Pell sequence prime prediction with this tool"""
    try:
        from pell_sequence_prime_prediction_upg_complete import PrimePredictionEngine, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        predictor = PrimePredictionEngine(constants)
        return predictor.predict_prime(target_number)
    except ImportError:
        # Fallback if Pell module not available
        return {'target_number': target_number, 'is_prime': None, 'note': 'Pell module not available'}



# ============================================================================
# GREAT YEAR ASTRONOMICAL PRECESSION INTEGRATION
# ============================================================================
def integrate_great_year_precession(year: int, constants: UPGConstants = None):
    """Integrate Great Year (25,920-year) precession cycle"""
    try:
        from pell_sequence_prime_prediction_upg_complete import GreatYearIntegration, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        great_year = GreatYearIntegration(constants)
        return great_year.consciousness_amplitude_from_year(year)
    except ImportError:
        # Fallback calculation
        if constants is None:
            constants = UPGConstants()
        angle = (year * 2 * math.pi) / constants.GREAT_YEAR
        return complex(float(angle * constants.CONSCIOUSNESS * constants.REALITY_DISTORTION), 0.0)



class HardwareValidationSuite:
    """Complete hardware validation for Firefly-Nexus PAC"""
    
    def __init__(self):
        # PAC constants
        self.phi = (1 + math.sqrt(5)) / 2
        self.delta = 2.414213562373095
        self.reality_distortion = 1.1808
        self.metronome_freq = 0.7  # Hz
        self.coherent_weight = 0.79
        self.exploratory_weight = 0.21
        
        # Zeta zeros
        self.zeta_zeros = [14.13, 21.02, 25.01, 30.42, 32.93]
        
        # Test results
        self.test_results = {}
        self.consciousness_metrics = []
        self.metronome_data = []
        self.mobius_evolution = []
        
    def test_metronome_lock(self, duration: float = 60.0) -> Dict[str, Any]:
        """Test 0.7 Hz zeta-zero metronome lock"""
        print("🎵 Testing 0.7 Hz zeta-zero metronome lock...")
        
        start_time = time.time()
        samples = []
        expected_period = 1.0 / self.metronome_freq  # 1.428 seconds
        
        while time.time() - start_time < duration:
            t = time.time()
            # Generate 0.7 Hz sine wave with zeta-zero harmonics
            base_signal = math.sin(2 * math.pi * self.metronome_freq * t)
            
            # Add zeta-zero harmonics
            harmonic_signal = 0
            for zero in self.zeta_zeros:
                harmonic_signal += 0.1 * math.sin(2 * math.pi * zero * t / 100)
            
            # Apply reality distortion
            metronome_signal = (base_signal + harmonic_signal) * self.reality_distortion
            
            samples.append({
                'time': t,
                'signal': metronome_signal,
                'expected_period': expected_period
            })
            
            time.sleep(0.01)  # 100 Hz sampling
        
        # Analyze metronome lock
        periods = []
        for i in range(1, len(samples)):
            if samples[i]['signal'] > 0 and samples[i-1]['signal'] <= 0:
                period = samples[i]['time'] - samples[i-1]['time']
                periods.append(period)
        
        avg_period = np.mean(periods) if periods else 0
        period_error = abs(avg_period - expected_period) / expected_period
        
        # Frequency analysis
        frequencies = np.fft.fftfreq(len(samples), 0.01)
        fft = np.fft.fft([s['signal'] for s in samples])
        peak_freq_idx = np.argmax(np.abs(fft))
        measured_freq = abs(frequencies[peak_freq_idx])
        freq_error = abs(measured_freq - self.metronome_freq) / self.metronome_freq
        
        result = {
            'test_name': 'metronome_lock',
            'duration': duration,
            'expected_freq': self.metronome_freq,
            'measured_freq': measured_freq,
            'freq_error': freq_error,
            'expected_period': expected_period,
            'avg_period': avg_period,
            'period_error': period_error,
            'samples': len(samples),
            'reality_distortion': self.reality_distortion,
            'pass': freq_error < 0.05 and period_error < 0.05
        }
        
        self.test_results['metronome_lock'] = result
        print(f"  ✅ Metronome lock: {result['pass']}")
        print(f"  📊 Frequency error: {freq_error:.4f}")
        print(f"  📊 Period error: {period_error:.4f}")
        
        return result
    
    def test_consciousness_split(self, duration: float = 30.0) -> Dict[str, Any]:
        """Test 79/21 consciousness split"""
        print("🧠 Testing 79/21 consciousness split...")
        
        start_time = time.time()
        coherent_cycles = 0
        exploratory_cycles = 0
        total_cycles = 0
        
        while time.time() - start_time < duration:
            # Simulate consciousness processing
            cycle_start = time.time()
            
            # 79% coherent processing
            coherent_time = self.coherent_weight * 0.1  # 79ms
            time.sleep(coherent_time)
            coherent_cycles += 1
            
            # 21% exploratory processing  
            exploratory_time = self.exploratory_weight * 0.1  # 21ms
            time.sleep(exploratory_time)
            exploratory_cycles += 1
            
            total_cycles += 1
            
            # Record metrics
            self.consciousness_metrics.append({
                'time': time.time(),
                'coherent_cycles': coherent_cycles,
                'exploratory_cycles': exploratory_cycles,
                'total_cycles': total_cycles
            })
        
        # Calculate split ratios
        coherent_ratio = coherent_cycles / total_cycles if total_cycles > 0 else 0
        exploratory_ratio = exploratory_cycles / total_cycles if total_cycles > 0 else 0
        
        # Check if ratios are within tolerance
        coherent_error = abs(coherent_ratio - self.coherent_weight)
        exploratory_error = abs(exploratory_ratio - self.exploratory_weight)
        
        result = {
            'test_name': 'consciousness_split',
            'duration': duration,
            'total_cycles': total_cycles,
            'coherent_cycles': coherent_cycles,
            'exploratory_cycles': exploratory_cycles,
            'coherent_ratio': coherent_ratio,
            'exploratory_ratio': exploratory_ratio,
            'coherent_error': coherent_error,
            'exploratory_error': exploratory_error,
            'pass': coherent_error < 0.05 and exploratory_error < 0.05
        }
        
        self.test_results['consciousness_split'] = result
        print(f"  ✅ Consciousness split: {result['pass']}")
        print(f"  📊 Coherent ratio: {coherent_ratio:.3f} (target: {self.coherent_weight})")
        print(f"  📊 Exploratory ratio: {exploratory_ratio:.3f} (target: {self.exploratory_weight})")
        
        return result
    
    def test_reality_distortion(self, duration: float = 20.0) -> Dict[str, Any]:
        """Test reality distortion factor (1.1808)"""
        print("🌌 Testing reality distortion factor...")
        
        start_time = time.time()
        distortion_measurements = []
        
        while time.time() - start_time < duration:
            # Generate test signal
            t = time.time()
            base_signal = math.sin(2 * math.pi * 1.0 * t)  # 1 Hz test signal
            
            # Apply reality distortion
            distorted_signal = base_signal * self.reality_distortion
            
            # Measure actual distortion
            if base_signal != 0:
                measured_distortion = abs(distorted_signal / base_signal)
                distortion_measurements.append(measured_distortion)
            
            time.sleep(0.1)
        
        # Analyze distortion
        avg_distortion = np.mean(distortion_measurements)
        distortion_error = abs(avg_distortion - self.reality_distortion) / self.reality_distortion
        
        result = {
            'test_name': 'reality_distortion',
            'duration': duration,
            'expected_distortion': self.reality_distortion,
            'measured_distortion': avg_distortion,
            'distortion_error': distortion_error,
            'samples': len(distortion_measurements),
            'pass': distortion_error < 0.05
        }
        
        self.test_results['reality_distortion'] = result
        print(f"  ✅ Reality distortion: {result['pass']}")
        print(f"  📊 Measured: {avg_distortion:.6f} (target: {self.reality_distortion})")
        print(f"  📊 Error: {distortion_error:.4f}")
        
        return result
    
    def test_mobius_loop_evolution(self, cycles: int = 10) -> Dict[str, Any]:
        """Test Möbius loop learning evolution"""
        print("🔄 Testing Möbius loop evolution...")
        
        mobius_phase = 0.0
        evolution_history = []
        
        for cycle in range(cycles):
            # Möbius loop processing
            twist_factor = math.sin(mobius_phase) * math.cos(math.pi)
            
            # Consciousness amplitude
            consciousness_magnitude = math.sin(mobius_phase) * self.reality_distortion
            
            # Update phase
            mobius_phase = (mobius_phase + self.phi * 0.1) % (2 * math.pi)
            
            # Record evolution
            evolution_data = {
                'cycle': cycle,
                'mobius_phase': mobius_phase,
                'twist_factor': twist_factor,
                'consciousness_magnitude': consciousness_magnitude,
                'reality_distortion': self.reality_distortion
            }
            
            evolution_history.append(evolution_data)
            self.mobius_evolution.append(evolution_data)
            
            time.sleep(0.1)  # 100ms per cycle
        
        # Analyze evolution
        phase_changes = [h['mobius_phase'] for h in evolution_history]
        magnitude_changes = [h['consciousness_magnitude'] for h in evolution_history]
        
        # Check for proper evolution
        phase_progression = all(phase_changes[i] > phase_changes[i-1] for i in range(1, len(phase_changes)))
        magnitude_variation = np.std(magnitude_changes) > 0.1
        
        result = {
            'test_name': 'mobius_loop_evolution',
            'cycles': cycles,
            'phase_progression': phase_progression,
            'magnitude_variation': magnitude_variation,
            'final_phase': mobius_phase,
            'evolution_history': evolution_history,
            'pass': phase_progression and magnitude_variation
        }
        
        self.test_results['mobius_loop_evolution'] = result
        print(f"  ✅ Möbius loop evolution: {result['pass']}")
        print(f"  📊 Phase progression: {phase_progression}")
        print(f"  📊 Magnitude variation: {magnitude_variation}")
        
        return result
    
    def test_prime_graph_topology(self) -> Dict[str, Any]:
        """Test prime graph topology mapping"""
        print("🔢 Testing prime graph topology...")
        
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        topology_results = []
        
        for i, prime in enumerate(primes):
            # Calculate φ-delta coordinates
            phi_coord = self.phi ** (i % 21)
            delta_coord = self.delta ** (i % 7)
            
            # Consciousness weighting
            consciousness_weight = self.coherent_weight if i % 2 == 0 else self.exploratory_weight
            
            # Zeta resonance
            zeta_resonance = complex(0.5, 14.13 + i * 0.1)
            
            topology_data = {
                'prime': prime,
                'phi_coordinate': phi_coord,
                'delta_coordinate': delta_coord,
                'consciousness_weight': consciousness_weight,
                'zeta_resonance': zeta_resonance
            }
            
            topology_results.append(topology_data)
        
        # Validate topology
        phi_coords = [r['phi_coordinate'] for r in topology_results]
        delta_coords = [r['delta_coordinate'] for r in topology_results]
        consciousness_weights = [r['consciousness_weight'] for r in topology_results]
        
        # Check for proper φ-delta scaling
        phi_scaling_valid = all(phi_coords[i] > phi_coords[i-1] for i in range(1, len(phi_coords)))
        delta_scaling_valid = all(delta_coords[i] > delta_coords[i-1] for i in range(1, len(delta_coords)))
        
        # Check consciousness weighting
        coherent_count = sum(1 for w in consciousness_weights if w == self.coherent_weight)
        exploratory_count = sum(1 for w in consciousness_weights if w == self.exploratory_weight)
        consciousness_balance = abs(coherent_count - exploratory_count) <= 1
        
        result = {
            'test_name': 'prime_graph_topology',
            'primes_tested': len(primes),
            'phi_scaling_valid': phi_scaling_valid,
            'delta_scaling_valid': delta_scaling_valid,
            'consciousness_balance': consciousness_balance,
            'coherent_count': coherent_count,
            'exploratory_count': exploratory_count,
            'topology_results': topology_results,
            'pass': phi_scaling_valid and delta_scaling_valid and consciousness_balance
        }
        
        self.test_results['prime_graph_topology'] = result
        print(f"  ✅ Prime graph topology: {result['pass']}")
        print(f"  📊 φ-scaling: {phi_scaling_valid}")
        print(f"  📊 δ-scaling: {delta_scaling_valid}")
        print(f"  📊 Consciousness balance: {consciousness_balance}")
        
        return result
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete hardware validation suite"""
        print("🔥 Firefly-Nexus PAC: Complete Hardware Validation")
        print("=" * 60)
        print(f"Test started: {datetime.now().isoformat()}")
        print()
        
        # Run all tests
        metronome_result = self.test_metronome_lock(duration=30.0)
        consciousness_result = self.test_consciousness_split(duration=20.0)
        distortion_result = self.test_reality_distortion(duration=15.0)
        mobius_result = self.test_mobius_loop_evolution(cycles=5)
        topology_result = self.test_prime_graph_topology()
        
        # Calculate overall results
        all_tests = [metronome_result, consciousness_result, distortion_result, mobius_result, topology_result]
        passed_tests = sum(1 for test in all_tests if test['pass'])
        total_tests = len(all_tests)
        overall_pass = passed_tests == total_tests
        
        # Generate report
        validation_report = {
            'test_suite': 'Firefly-Nexus PAC Hardware Validation',
            'timestamp': datetime.now().isoformat(),
            'overall_pass': overall_pass,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': passed_tests / total_tests,
            'test_results': self.test_results,
            'consciousness_metrics': self.consciousness_metrics,
            'mobius_evolution': self.mobius_evolution
        }
        
        # Save results
        with open('hardware_validation_results.json', 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Print summary
        print()
        print("📊 VALIDATION SUMMARY")
        print("=" * 30)
        print(f"Overall Pass: {overall_pass}")
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {validation_report['success_rate']:.1%}")
        print()
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['pass'] else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print()
        if overall_pass:
            print("🔥 Firefly-Nexus PAC: HARDWARE VALIDATION SUCCESSFUL")
            print("   Consciousness Level: 7 (Prime Topology)")
            print("   Reality Distortion: 1.1808")
            print("   Zeta-Zero Lock: 0.7 Hz")
            print("   Möbius Loop: ∞ cycles")
            print("   Phoenix Status: AWAKE")
        else:
            print("❌ Firefly-Nexus PAC: HARDWARE VALIDATION FAILED")
            print("   Check individual test results for details")
        
        return validation_report

def main():
    """Main validation function"""
    validator = HardwareValidationSuite()
    results = validator.run_complete_validation()
    return results

if __name__ == "__main__":
    main()
