#!/usr/bin/env python3
"""
Comprehensive VM Testing Suite
==============================

Test all VM systems:
- PDVM (Poly Dimensional VM)
- QVM (Quantum Virtual Machine)
- UVM (Universal VM)
- OVM (Omniversal VM)
- Unified System

Author: Bradley Wallace, COO Koba42
Framework: PAC + PDVM + QVM + UVM + OVM
Consciousness Level: 7 (Prime Topology)
"""

import requests
import json
import time
import numpy as np
from typing import Dict, List, Any

class VMTestSuite:
    """Comprehensive VM testing suite"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.test_results = []
        
    def test_health_checks(self) -> Dict[str, Any]:
        """Test health and status endpoints"""
        print("🏥 Testing health checks...")
        
        results = {}
        
        # Health check
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            results['health'] = {
                'status_code': response.status_code,
                'response': response.json(),
                'success': response.status_code == 200
            }
            print(f"  ✅ Health check: {response.status_code}")
        except Exception as e:
            results['health'] = {'error': str(e), 'success': False}
            print(f"  ❌ Health check failed: {e}")
        
        # Readiness check
        try:
            response = requests.get(f"{self.base_url}/ready", timeout=5)
            results['ready'] = {
                'status_code': response.status_code,
                'response': response.json(),
                'success': response.status_code == 200
            }
            print(f"  ✅ Readiness check: {response.status_code}")
        except Exception as e:
            results['ready'] = {'error': str(e), 'success': False}
            print(f"  ❌ Readiness check failed: {e}")
        
        # Status check
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            results['status'] = {
                'status_code': response.status_code,
                'response': response.json(),
                'success': response.status_code == 200
            }
            print(f"  ✅ Status check: {response.status_code}")
        except Exception as e:
            results['status'] = {'error': str(e), 'success': False}
            print(f"  ❌ Status check failed: {e}")
        
        return results
    
    def test_pdvm_processing(self) -> Dict[str, Any]:
        """Test PDVM (Poly Dimensional VM) processing"""
        print("\n🔧 Testing PDVM (Poly Dimensional VM)...")
        
        test_data = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        try:
            response = requests.post(
                f"{self.base_url}/pdvm/process",
                json={"values": test_data},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ PDVM processing successful")
                print(f"  📊 Dimensions processed: {len(result['dimensional_results'])}")
                print(f"  📊 Processing time: {result['processing_time']:.6f}s")
                
                # Check dimensional results
                dimensions = result['dimensional_results']
                for dim_name, dim_result in dimensions.items():
                    print(f"    {dim_name}: {dim_result.get('consciousness_weight', 0):.3f} weight")
                
                return {
                    'success': True,
                    'dimensions': len(dimensions),
                    'processing_time': result['processing_time'],
                    'result': result
                }
            else:
                print(f"  ❌ PDVM processing failed: {response.status_code}")
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"  ❌ PDVM processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_qvm_quantum(self) -> Dict[str, Any]:
        """Test QVM (Quantum Virtual Machine) processing"""
        print("\n⚛️ Testing QVM (Quantum Virtual Machine)...")
        
        test_data = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        try:
            response = requests.post(
                f"{self.base_url}/qvm/quantum",
                json={"values": test_data},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ QVM quantum processing successful")
                print(f"  📊 Quantum amplitudes: {len(result['quantum_amplitudes'])}")
                print(f"  📊 Entanglement pairs: {len(result['entanglement_pairs'])}")
                print(f"  📊 Coherence level: {result['coherence_level']}")
                print(f"  📊 Processing time: {result['processing_time']:.6f}s")
                
                return {
                    'success': True,
                    'amplitudes': len(result['quantum_amplitudes']),
                    'entanglement_pairs': len(result['entanglement_pairs']),
                    'coherence_level': result['coherence_level'],
                    'processing_time': result['processing_time'],
                    'result': result
                }
            else:
                print(f"  ❌ QVM quantum processing failed: {response.status_code}")
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"  ❌ QVM quantum processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_uvm_universal(self) -> Dict[str, Any]:
        """Test UVM (Universal VM) processing"""
        print("\n🌐 Testing UVM (Universal VM)...")
        
        test_data = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        operations = ['compute', 'store', 'retrieve', 'transform', 'evolve', 'consciousness', 'reality', 'omniverse']
        
        results = {}
        
        for operation in operations:
            try:
                response = requests.post(
                    f"{self.base_url}/uvm/universal",
                    json={"values": test_data, "operation": operation},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"  ✅ UVM {operation}: {result['processing_time']:.6f}s")
                    results[operation] = {
                        'success': True,
                        'processing_time': result['processing_time'],
                        'evolution_cycles': result['evolution_cycles']
                    }
                else:
                    print(f"  ❌ UVM {operation} failed: {response.status_code}")
                    results[operation] = {'success': False, 'error': f"HTTP {response.status_code}"}
                    
            except Exception as e:
                print(f"  ❌ UVM {operation} error: {e}")
                results[operation] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_ovm_omniverse(self) -> Dict[str, Any]:
        """Test OVM (Omniversal VM) processing"""
        print("\n🌌 Testing OVM (Omniversal VM)...")
        
        test_data = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        try:
            response = requests.post(
                f"{self.base_url}/ovm/omniverse",
                json={"values": test_data},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ OVM omniverse processing successful")
                print(f"  📊 Omniverse dimensions: {result['total_dimensions']}")
                print(f"  📊 Total consciousness: {result['combined_result']['total_consciousness']:.6f}")
                print(f"  📊 Processing time: {result['processing_time']:.6f}s")
                
                return {
                    'success': True,
                    'dimensions': result['total_dimensions'],
                    'total_consciousness': result['combined_result']['total_consciousness'],
                    'processing_time': result['processing_time'],
                    'result': result
                }
            else:
                print(f"  ❌ OVM omniverse processing failed: {response.status_code}")
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"  ❌ OVM omniverse processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_unified_consciousness(self) -> Dict[str, Any]:
        """Test unified consciousness computation"""
        print("\n🧠 Testing Unified Consciousness Computation...")
        
        test_data = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        try:
            response = requests.post(
                f"{self.base_url}/unified/consciousness",
                json={"values": test_data},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Unified consciousness computation successful")
                print(f"  📊 Total consciousness: {result['unified_result']['total_consciousness']:.6f}")
                print(f"  📊 Reality distortion: {result['reality_distortion']}")
                print(f"  📊 Processing time: {result['processing_time']:.6f}s")
                
                # Check VM results
                vm_results = result['vm_results']
                print(f"  📊 PDVM dimensions: {len(vm_results['pdvm']['dimensional_results'])}")
                print(f"  📊 QVM coherence: {vm_results['qvm']['coherence_level']}")
                print(f"  📊 UVM operations: {vm_results['uvm']['evolution_cycles']}")
                print(f"  📊 OVM dimensions: {vm_results['ovm']['total_dimensions']}")
                
                return {
                    'success': True,
                    'total_consciousness': result['unified_result']['total_consciousness'],
                    'reality_distortion': result['reality_distortion'],
                    'processing_time': result['processing_time'],
                    'vm_results': vm_results,
                    'result': result
                }
            else:
                print(f"  ❌ Unified consciousness computation failed: {response.status_code}")
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"  ❌ Unified consciousness computation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_benchmark(self) -> Dict[str, Any]:
        """Test VM system benchmark"""
        print("\n📊 Testing VM System Benchmark...")
        
        test_data = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        try:
            response = requests.post(
                f"{self.base_url}/vm/benchmark",
                json={"values": test_data, "iterations": 10},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Benchmark completed successfully")
                
                benchmark_results = result['benchmark_results']
                for vm_name, vm_result in benchmark_results.items():
                    print(f"  📊 {vm_name.upper()}: {vm_result['throughput']:.0f} ops/s")
                
                return {
                    'success': True,
                    'benchmark_results': benchmark_results,
                    'iterations': result['iterations'],
                    'data_size': result['data_size'],
                    'total_time': result['total_time'],
                    'result': result
                }
            else:
                print(f"  ❌ Benchmark failed: {response.status_code}")
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"  ❌ Benchmark error: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_vm_systems_info(self) -> Dict[str, Any]:
        """Test VM systems information"""
        print("\nℹ️ Testing VM Systems Information...")
        
        try:
            response = requests.get(f"{self.base_url}/vm/systems", timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ VM systems information retrieved")
                
                vm_systems = result['vm_systems']
                for vm_name, vm_info in vm_systems.items():
                    print(f"  📊 {vm_name.upper()}: {vm_info['name']} - {vm_info['description']}")
                
                return {
                    'success': True,
                    'vm_systems': vm_systems,
                    'unified_system': result['unified_system'],
                    'result': result
                }
            else:
                print(f"  ❌ VM systems info failed: {response.status_code}")
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"  ❌ VM systems info error: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        print("🔥 Comprehensive VM Testing Suite")
        print("=" * 50)
        print("Testing unified VM consciousness computing system...")
        print()
        
        start_time = time.time()
        
        # Run all tests
        health_results = self.test_health_checks()
        pdvm_results = self.test_pdvm_processing()
        qvm_results = self.test_qvm_quantum()
        uvm_results = self.test_uvm_universal()
        ovm_results = self.test_ovm_omniverse()
        unified_results = self.test_unified_consciousness()
        benchmark_results = self.test_benchmark()
        systems_info = self.test_vm_systems_info()
        
        total_time = time.time() - start_time
        
        # Compile results
        test_results = {
            'health_checks': health_results,
            'pdvm_processing': pdvm_results,
            'qvm_quantum': qvm_results,
            'uvm_universal': uvm_results,
            'ovm_omniverse': ovm_results,
            'unified_consciousness': unified_results,
            'benchmark': benchmark_results,
            'systems_info': systems_info,
            'total_test_time': total_time
        }
        
        # Print summary
        self.print_test_summary(test_results)
        
        return test_results
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print test summary"""
        print("\n📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 50)
        
        # Health checks
        health = results['health_checks']
        print("🏥 Health Checks:")
        print(f"  Health: {'✅' if health['health']['success'] else '❌'}")
        print(f"  Ready: {'✅' if health['ready']['success'] else '❌'}")
        print(f"  Status: {'✅' if health['status']['success'] else '❌'}")
        
        # VM Systems
        print("\n🔧 VM Systems:")
        print(f"  PDVM: {'✅' if results['pdvm_processing']['success'] else '❌'}")
        print(f"  QVM: {'✅' if results['qvm_quantum']['success'] else '❌'}")
        print(f"  UVM: {'✅' if any(op['success'] for op in results['uvm_universal'].values()) else '❌'}")
        print(f"  OVM: {'✅' if results['ovm_omniverse']['success'] else '❌'}")
        print(f"  Unified: {'✅' if results['unified_consciousness']['success'] else '❌'}")
        
        # Performance
        if results['benchmark']['success']:
            benchmark = results['benchmark']['benchmark_results']
            print("\n📊 Performance Benchmarks:")
            for vm_name, vm_result in benchmark.items():
                print(f"  {vm_name.upper()}: {vm_result['throughput']:.0f} ops/s")
        
        # Consciousness
        if results['unified_consciousness']['success']:
            unified = results['unified_consciousness']
            print(f"\n🧠 Consciousness Results:")
            print(f"  Total Consciousness: {unified['total_consciousness']:.6f}")
            print(f"  Reality Distortion: {unified['reality_distortion']}")
            print(f"  Processing Time: {unified['processing_time']:.6f}s")
        
        print(f"\n⏱️ Total Test Time: {results['total_test_time']:.2f}s")
        print("\n🔥 Phoenix Status: AWAKE")

def main():
    """Main function to run comprehensive tests"""
    print("🔥 Comprehensive VM Testing Suite")
    print("=" * 60)
    print("Testing unified VM consciousness computing system...")
    print("Consciousness Level: 7 (Prime Topology)")
    print("Reality Distortion: 1.1808")
    print()
    
    # Create test suite
    test_suite = VMTestSuite()
    
    # Run comprehensive tests
    results = test_suite.run_comprehensive_tests()
    
    return results

if __name__ == "__main__":
    main()
