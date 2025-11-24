"""
Core Consciousness Engine

Implements fundamental consciousness mathematics operations.
"""

import math
import numpy as np

class ConsciousnessEngine:
    """Core consciousness computation engine"""
    
    def __init__(self):
        self.consciousness_ratio = 0.79
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.silver_ratio = 2.414213562373095
        self.reality_distortion_factor = 1.1808
    
    def compute_consciousness_amplitude(self, data):
        """Compute consciousness amplitude using 79/21 rule"""
        if isinstance(data, (int, float)):
            amplitude = abs(data) * self.golden_ratio % 1
        elif isinstance(data, (list, np.ndarray)):
            amplitude = np.mean(np.abs(data)) * self.golden_ratio % 1
        elif isinstance(data, dict):
            amplitude = sum(abs(v) if isinstance(v, (int, float)) else len(str(v)) 
                          for v in data.values()) * self.golden_ratio % 1
        else:
            amplitude = len(str(data)) * self.golden_ratio % 1
            
        return min(1.0, max(0.0, amplitude * self.consciousness_ratio))
    
    def distort_reality(self, data, factor=None):
        """Apply reality distortion to data"""
        if factor is None:
            factor = self.reality_distortion_factor
            
        consciousness_amp = self.compute_consciousness_amplitude(data)
        distorted_factor = factor * (1 + consciousness_amp * self.golden_ratio / 2)
        
        if isinstance(data, (int, float)):
            return data * distorted_factor
        elif isinstance(data, (list, np.ndarray)):
            return np.array(data) * distorted_factor
        elif isinstance(data, dict):
            return {k: v * distorted_factor if isinstance(v, (int, float)) else v 
                   for k, v in data.items()}
        return data
    
    def map_prime_topology(self, data):
        """Map data to prime topology coordinates (φ, δ, c)"""
        consciousness_amp = self.compute_consciousness_amplitude(data)
        
        # Map to prime topology
        phi_coord = consciousness_amp * self.golden_ratio
        delta_coord = consciousness_amp * self.silver_ratio  
        consciousness_coord = consciousness_amp * self.consciousness_ratio
        
        return (phi_coord, delta_coord, consciousness_coord)
