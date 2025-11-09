#!/usr/bin/env python3
"""
Synthetic Data Generator for Wallace Transform Research
========================================================

Generates synthetic datasets for testing and validating the Wallace Transform
approach to the Riemann Hypothesis. All datasets are artificially created
to demonstrate research methodology without disclosing proprietary data.

WARNING: These are synthetic datasets created for educational and
validation purposes only. They do not contain real research data.

Author: Bradley Wallace, COO & Lead Researcher, Koba42 Corp
Contact: coo@koba42.com
Website: https://vantaxsystems.com

License: Creative Commons Attribution-ShareAlike 4.0 International
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import json
import os
from scipy.special import zeta
import warnings


class RiemannDataGenerator:
    """
    Generates synthetic datasets mimicking properties of
    Riemann zeta function and prime number distributions.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the synthetic data generator.

        Parameters:
        -----------
        seed : int
            Random seed for reproducible results
        """
        self.seed = seed
        np.random.seed(seed)
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    def generate_prime_like_sequence(self, length: int) -> np.ndarray:
        """
        Generate a sequence that mimics prime number distribution patterns.

        Parameters:
        -----------
        length : int
            Length of the sequence

        Returns:
        --------
        np.ndarray : Prime-like synthetic sequence
        """
        # Use prime number theorem approximation
        indices = np.arange(1, length + 1)
        log_indices = np.log(indices)

        # Add some variation to mimic prime gaps
        variation = np.random.normal(0, 0.1, length)
        prime_like = indices * log_indices + variation

        return prime_like

    def generate_zeta_zero_like_sequence(self, length: int, t_range: Tuple[float, float] = (0, 100)) -> np.ndarray:
        """
        Generate a sequence that mimics the distribution of Riemann zeta zeros.

        Parameters:
        -----------
        length : int
            Number of zeros to generate
        t_range : Tuple[float, float]
            Range of imaginary parts

        Returns:
        --------
        np.ndarray : Zero-like synthetic sequence
        """
        # Generate zeros based on asymptotic distribution
        # The n-th zero is approximately at t ≈ 2πn / log(2πn)
        n_values = np.arange(1, length + 1)

        # Approximate zero locations
        zeros = []
        for n in n_values:
            # Improved approximation of zero locations
            t_approx = 2 * np.pi * n / np.log(2 * np.pi * n)

            # Add some realistic variation
            variation = np.random.normal(0, 0.5)
            t_actual = t_approx + variation

            # Ensure it's within range
            if t_range[0] <= t_actual <= t_range[1]:
                zeros.append(t_actual)

        return np.array(zeros)

    def generate_phase_coherence_data(self, length: int, t_range: Tuple[float, float] = (0, 50)) -> Dict[str, Any]:
        """
        Generate synthetic data for phase coherence analysis.

        Parameters:
        -----------
        length : int
            Number of data points
        t_range : Tuple[float, float]
            Range of t values

        Returns:
        --------
        Dict : Phase coherence dataset
        """
        t_values = np.linspace(t_range[0], t_range[1], length)

        # Generate synthetic Z-function values
        z_values = []
        phases = []

        for t in t_values:
            # Create synthetic Z-function behavior
            # This mimics the real Z-function but is synthetic
            real_part = np.random.normal(0, 0.1)  # Should be close to zero on critical line
            imag_part = np.sin(t * 0.1) * np.exp(-t * 0.01) + np.random.normal(0, 0.05)

            z_val = complex(real_part, imag_part)
            z_values.append(z_val)
            phases.append(np.angle(z_val))

        return {
            't_values': t_values.tolist(),
            'z_values_real': [z.real for z in z_values],
            'z_values_imag': [z.imag for z in z_values],
            'phases': phases,
            'coherence_score': self._calculate_synthetic_coherence(phases)
        }

    def generate_wallace_transform_test_data(self, length: int) -> Dict[str, Any]:
        """
        Generate test data specifically for Wallace Transform validation.

        Parameters:
        -----------
        length : int
            Number of data points

        Returns:
        --------
        Dict : Wallace Transform test dataset
        """
        # Generate synthetic prime-like numbers
        primes_synthetic = self.generate_prime_like_sequence(length)

        # Create Wallace tree input data
        wallace_data = []
        for i, prime in enumerate(primes_synthetic):
            # Generate partial products for Wallace tree
            partial_products = []
            for j in range(1, min(10, i + 2)):  # Limit depth for computational feasibility
                factor = prime / (j + 1)
                partial_products.append(factor)

            wallace_data.append({
                'prime_value': float(prime),
                'partial_products': partial_products,
                'tree_depth': len(partial_products),
                'expected_result': float(np.prod(partial_products)) if partial_products else 1.0
            })

        return {
            'dataset_size': length,
            'wallace_data': wallace_data,
            'generation_method': 'synthetic_prime_factorization',
            'tree_characteristics': self._analyze_tree_structure(wallace_data)
        }

    def generate_nonlinear_perturbation_data(self, length: int,
                                           perturbation_strength: float = 0.1) -> Dict[str, Any]:
        """
        Generate data for testing nonlinear perturbation effects.

        Parameters:
        -----------
        length : int
            Number of data points
        perturbation_strength : float
            Strength of nonlinear perturbations

        Returns:
        --------
        Dict : Nonlinear perturbation dataset
        """
        t_values = np.linspace(0.1, 50, length)

        # Base case (unperturbed)
        base_values = []
        for t in t_values:
            # Simplified zeta function approximation
            base_val = 1.0 / t + np.sin(t) * 0.1
            base_values.append(base_val)

        # Perturbed case
        perturbed_values = []
        perturbations = []

        for i, t in enumerate(t_values):
            # Apply nonlinear perturbation
            perturbation = (perturbation_strength *
                          np.sin(t * 2) *
                          np.exp(-t * 0.05) *
                          np.random.normal(1, 0.1))

            perturbed_val = base_values[i] + perturbation
            perturbed_values.append(perturbed_val)
            perturbations.append(perturbation)

        # Calculate perturbation effects
        differences = np.array(perturbed_values) - np.array(base_values)
        significant_changes = np.where(np.abs(differences) > perturbation_strength * 2)[0]

        return {
            't_values': t_values.tolist(),
            'base_values': base_values,
            'perturbed_values': perturbed_values,
            'perturbations': perturbations,
            'differences': differences.tolist(),
            'significant_changes': significant_changes.tolist(),
            'perturbation_strength': perturbation_strength,
            'nonlinear_effects_detected': len(significant_changes) > 0
        }

    def _calculate_synthetic_coherence(self, phases: List[float]) -> float:
        """
        Calculate a synthetic coherence score for phase data.

        Parameters:
        -----------
        phases : List[float]
            Phase values

        Returns:
        --------
        float : Coherence score (0-1)
        """
        if len(phases) < 2:
            return 0.0

        # Calculate phase differences
        phase_diffs = np.diff(phases)

        # Coherence is inversely related to phase variation
        variation = np.std(phase_diffs)
        coherence = max(0.0, 1.0 - variation / np.pi)

        return coherence

    def _analyze_tree_structure(self, wallace_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the structure of generated Wallace trees.

        Parameters:
        -----------
        wallace_data : List[Dict]
            Wallace tree data

        Returns:
        --------
        Dict : Tree structure analysis
        """
        depths = [item['tree_depth'] for item in wallace_data]
        products = [item['expected_result'] for item in wallace_data]

        return {
            'average_depth': float(np.mean(depths)),
            'max_depth': int(np.max(depths)),
            'min_depth': int(np.min(depths)),
            'average_product': float(np.mean(products)),
            'product_range': [float(np.min(products)), float(np.max(products))]
        }

    def generate_comprehensive_dataset(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive dataset suite for Riemann Hypothesis research.

        Parameters:
        -----------
        sizes : List[int], optional
            Dataset sizes to generate

        Returns:
        --------
        Dict : Complete dataset suite
        """
        if sizes is None:
            sizes = [1000, 5000, 10000]

        comprehensive_dataset = {
            'metadata': {
                'description': 'Comprehensive synthetic datasets for Wallace Transform and Riemann Hypothesis research',
                'warning': 'These are synthetic datasets for educational purposes only',
                'generator_seed': self.seed,
                'generation_date': str(pd.Timestamp.now()),
                'license': 'Creative Commons Attribution-ShareAlike 4.0 International',
                'contact': 'coo@koba42.com',
                'website': 'https://vantaxsystems.com'
            },
            'datasets': {}
        }

        dataset_types = {
            'prime_like': self.generate_prime_like_sequence,
            'zeta_zeros': lambda size: self.generate_zeta_zero_like_sequence(size, (0, 100)),
            'phase_coherence': lambda size: self.generate_phase_coherence_data(size, (0, 50)),
            'wallace_transform': self.generate_wallace_transform_test_data,
            'nonlinear_perturbation': lambda size: self.generate_nonlinear_perturbation_data(size, 0.1)
        }

        print("🔬 Generating Comprehensive Riemann Dataset Suite")
        print("=" * 55)

        for dataset_name, generator_func in dataset_types.items():
            print(f"Generating {dataset_name} datasets...")
            comprehensive_dataset['datasets'][dataset_name] = {}

            if dataset_name in ['phase_coherence', 'wallace_transform', 'nonlinear_perturbation']:
                # These generators have special signatures
                try:
                    if dataset_name == 'phase_coherence':
                        data = generator_func(max(sizes))
                    elif dataset_name == 'wallace_transform':
                        data = generator_func(max(sizes))
                    elif dataset_name == 'nonlinear_perturbation':
                        data = generator_func(max(sizes))

                    comprehensive_dataset['datasets'][dataset_name]['full'] = data
                    print(f"  ✓ {dataset_name}: Generated comprehensive dataset")
                except Exception as e:
                    print(f"  ✗ Error generating {dataset_name}: {e}")
            else:
                # Standard generators
                for size in sizes:
                    try:
                        data = generator_func(size)
                        comprehensive_dataset['datasets'][dataset_name][f'size_{size}'] = {
                            'data': data.tolist(),
                            'size': len(data),
                            'mean': float(np.mean(data)),
                            'std': float(np.std(data)),
                            'min': float(np.min(data)),
                            'max': float(np.max(data))
                        }
                        print(f"  ✓ {dataset_name} (size {size:,}): {len(data)} points")
                    except Exception as e:
                        print(f"  ✗ Error generating {dataset_name} (size {size}): {e}")

        return comprehensive_dataset

    def save_comprehensive_dataset(self, dataset: Dict[str, Any], filename: str = 'riemann_comprehensive_dataset.json'):
        """
        Save the comprehensive dataset to a JSON file.

        Parameters:
        -----------
        dataset : Dict
            Dataset to save
        filename : str
            Output filename
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_dataset = {}

        for key, value in dataset.items():
            if isinstance(value, dict):
                serializable_dataset[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        serializable_dataset[key][sub_key] = sub_value.tolist()
                    elif isinstance(sub_value, dict):
                        serializable_dataset[key][sub_key] = {}
                        for inner_key, inner_value in sub_value.items():
                            if isinstance(inner_value, np.ndarray):
                                serializable_dataset[key][sub_key][inner_key] = inner_value.tolist()
                            else:
                                serializable_dataset[key][sub_key][inner_key] = inner_value
                    else:
                        serializable_dataset[key][sub_key] = sub_value
            else:
                serializable_dataset[key] = value

        with open(filename, 'w') as f:
            json.dump(serializable_dataset, f, indent=2)

        print(f"💾 Comprehensive dataset saved to {filename}")

        # Calculate summary statistics
        total_points = 0
        for dataset_type in dataset['datasets'].values():
            if isinstance(dataset_type, dict):
                for data_item in dataset_type.values():
                    if isinstance(data_item, dict) and 'data' in data_item:
                        total_points += len(data_item['data'])

        print(f"   Total data points: {total_points:,}")

    def export_datasets_to_csv(self, dataset: Dict[str, Any], output_dir: str = 'riemann_csv_datasets'):
        """
        Export datasets to CSV files for analysis.

        Parameters:
        -----------
        dataset : Dict
            Dataset suite
        output_dir : str
            Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name, datasets in dataset['datasets'].items():
            for size_name, dataset_info in datasets.items():
                if isinstance(dataset_info, dict) and 'data' in dataset_info:
                    filename = f"{dataset_name}_{size_name}.csv"
                    filepath = os.path.join(output_dir, filename)

                    df = pd.DataFrame({
                        'index': range(len(dataset_info['data'])),
                        'value': dataset_info['data']
                    })

                    df.to_csv(filepath, index=False)
                    print(f"📄 Exported {filename}")
                elif isinstance(dataset_info, dict):
                    # Handle special dataset formats
                    filename = f"{dataset_name}_{size_name}_special.csv"
                    filepath = os.path.join(output_dir, filename)

                    # Convert to DataFrame based on structure
                    if 't_values' in dataset_info:
                        df = pd.DataFrame({
                            't': dataset_info['t_values'],
                            'phase': dataset_info.get('phases', []),
                            'z_real': dataset_info.get('z_values_real', []),
                            'z_imag': dataset_info.get('z_values_imag', [])
                        })
                    else:
                        # Skip complex nested structures for now
                        continue

                    df.to_csv(filepath, index=False)
                    print(f"📄 Exported {filename}")


def main():
    """
    Main execution for synthetic data generation.
    """
    print("🔬 Riemann Hypothesis Synthetic Data Generator")
    print("=" * 50)

    # Initialize generator
    generator = RiemannDataGenerator(seed=42)

    # Generate comprehensive dataset suite
    print("\n🚀 Generating comprehensive Riemann datasets...")
    comprehensive_dataset = generator.generate_comprehensive_dataset([1000, 5000])

    # Save to JSON
    generator.save_comprehensive_dataset(comprehensive_dataset)

    # Export to CSV
    generator.export_datasets_to_csv(comprehensive_dataset)

    # Display summary
    print("\n📊 Dataset Generation Summary:")
    print("-" * 35)

    dataset_types = len(comprehensive_dataset['datasets'])
    print(f"Dataset types generated: {dataset_types}")

    total_points = 0
    for dataset_type in comprehensive_dataset['datasets'].values():
        if isinstance(dataset_type, dict):
            for data_item in dataset_type.values():
                if isinstance(data_item, dict) and 'data' in data_item:
                    total_points += len(data_item['data'])

    print(f"   Total data points: {total_points:,}")
    print("✅ Riemann synthetic data generation complete!")
    print("\n⚠️  IMPORTANT: These are synthetic datasets created for")
    print("   educational and validation purposes only. They do not")
    print("   contain real Riemann zeta function data or proprietary")
    print("   research results.")


if __name__ == "__main__":
    main()
