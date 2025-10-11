#!/usr/bin/env python3
"""
CUDNT Model Quantization & Pruning - Edge Deployment Optimization
================================================================

Automatic model compression and quantization for efficient edge deployment.
Reduces model size and inference time while maintaining accuracy.
"""

import numpy as np
import copy
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import json
import os

logger = logging.getLogger(__name__)

class ModelQuantizer:
    """
    Model quantization for reduced precision inference.
    Supports 8-bit, 4-bit, and dynamic quantization.
    """

    def __init__(self, quantization_config: Optional[Dict[str, Any]] = None):
        """Initialize quantizer."""
        self.config = quantization_config or self._default_config()
        self.quantization_stats = {}

        logger.info("🔢 Model Quantizer initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default quantization configuration."""
        return {
            'precision': 'int8',  # int8, int4, dynamic
            'calibration_samples': 1000,
            'per_channel': True,
            'symmetric': True,
            'enable_bias_correction': True
        }

    def quantize_model(self, model: 'CUDNT_Model', calibration_data: Optional[np.ndarray] = None) -> 'QuantizedModel':
        """Quantize model for reduced precision inference."""
        logger.info(f"Quantizing model with {self.config['precision']} precision...")

        # Create calibration data if not provided
        if calibration_data is None:
            calibration_data = self._generate_calibration_data(model)

        # Quantize each layer
        quantized_parameters = {}
        quantization_info = {}

        for layer_name, param_tensor in model.parameters.items():
            quantized_param, quant_info = self._quantize_parameter(
                param_tensor, layer_name, calibration_data
            )
            quantized_parameters[layer_name] = quantized_param
            quantization_info[layer_name] = quant_info

        # Create quantized model
        quantized_model = QuantizedModel(
            model.architecture,
            quantized_parameters,
            quantization_info,
            self.config
        )

        self.quantization_stats = {
            'original_size_mb': self._calculate_model_size(model.parameters),
            'quantized_size_mb': self._calculate_model_size(quantized_parameters),
            'compression_ratio': self._calculate_compression_ratio(model.parameters, quantized_parameters),
            'quantization_info': quantization_info
        }

        logger.info(".1f"        return quantized_model

    def _generate_calibration_data(self, model: 'CUDNT_Model', n_samples: int = 1000) -> np.ndarray:
        """Generate calibration data for quantization."""
        # Create synthetic data similar to expected input distribution
        if hasattr(model, '_get_input_shape'):
            input_shape = model._get_input_shape()
            return np.random.randn(n_samples, *input_shape[1:]).astype(np.float32)
        else:
            # Default: assume 784 features (like MNIST flattened)
            return np.random.randn(n_samples, 784).astype(np.float32)

    def _quantize_parameter(self, param_tensor: np.ndarray, layer_name: str,
                          calibration_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize a single parameter tensor."""
        if self.config['precision'] == 'int8':
            return self._quantize_int8(param_tensor, layer_name)
        elif self.config['precision'] == 'int4':
            return self._quantize_int4(param_tensor, layer_name)
        elif self.config['precision'] == 'dynamic':
            return self._quantize_dynamic(param_tensor, layer_name)
        else:
            raise ValueError(f"Unsupported precision: {self.config['precision']}")

    def _quantize_int8(self, tensor: np.ndarray, layer_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to 8-bit integers."""
        # Calculate scale and zero point
        if self.config['symmetric']:
            # Symmetric quantization: [-127, 127]
            abs_max = np.max(np.abs(tensor))
            scale = abs_max / 127.0
            zero_point = 0
        else:
            # Asymmetric quantization: [0, 255]
            tensor_min = np.min(tensor)
            tensor_max = np.max(tensor)
            scale = (tensor_max - tensor_min) / 255.0
            zero_point = np.round(-tensor_min / scale)

        # Quantize
        quantized = np.round(tensor / scale + zero_point).astype(np.int8)

        # Apply bias correction if enabled
        if self.config['enable_bias_correction']:
            quantized_float = (quantized.astype(np.float32) - zero_point) * scale
            correction = tensor - quantized_float
            quantized = np.round((tensor - correction.mean()) / scale + zero_point).astype(np.int8)

        quant_info = {
            'scale': float(scale),
            'zero_point': int(zero_point),
            'original_dtype': str(tensor.dtype),
            'quantized_dtype': 'int8',
            'compression_ratio': 4.0  # float32 -> int8
        }

        return quantized, quant_info

    def _quantize_int4(self, tensor: np.ndarray, layer_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to 4-bit integers."""
        # Similar to int8 but with 4-bit range [-8, 7]
        abs_max = np.max(np.abs(tensor))
        scale = abs_max / 8.0

        # Quantize to 4-bit values
        quantized = np.round(tensor / scale).astype(np.int8)
        quantized = np.clip(quantized, -8, 7)  # 4-bit range

        quant_info = {
            'scale': float(scale),
            'zero_point': 0,
            'original_dtype': str(tensor.dtype),
            'quantized_dtype': 'int4',
            'compression_ratio': 8.0  # float32 -> int4
        }

        return quantized, quant_info

    def _quantize_dynamic(self, tensor: np.ndarray, layer_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dynamic quantization (quantize at runtime)."""
        # For dynamic quantization, we keep the original tensor but prepare quantization parameters
        abs_max = np.max(np.abs(tensor))
        scale = abs_max / 127.0

        quant_info = {
            'scale': float(scale),
            'zero_point': 0,
            'original_dtype': str(tensor.dtype),
            'quantized_dtype': 'dynamic_int8',
            'compression_ratio': 1.0  # No compression, quantized at runtime
        }

        return tensor, quant_info

    def _calculate_model_size(self, parameters: Dict[str, np.ndarray]) -> float:
        """Calculate total model size in MB."""
        total_bytes = sum(tensor.nbytes for tensor in parameters.values())
        return total_bytes / (1024 * 1024)

    def _calculate_compression_ratio(self, original_params: Dict[str, np.ndarray],
                                   quantized_params: Dict[str, np.ndarray]) -> float:
        """Calculate compression ratio."""
        original_size = sum(tensor.nbytes for tensor in original_params.values())
        quantized_size = sum(tensor.nbytes for tensor in quantized_params.values())
        return original_size / quantized_size


class ModelPruner:
    """
    Model pruning for reduced complexity and size.
    Supports magnitude-based, structured, and dynamic pruning.
    """

    def __init__(self, pruning_config: Optional[Dict[str, Any]] = None):
        """Initialize pruner."""
        self.config = pruning_config or self._default_config()
        self.pruning_stats = {}

        logger.info("✂️ Model Pruner initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default pruning configuration."""
        return {
            'method': 'magnitude',  # magnitude, structured, l1
            'sparsity': 0.5,  # Target sparsity (0.5 = 50% pruned)
            'schedule': 'one_shot',  # one_shot, gradual, dynamic
            'granularity': 'element',  # element, filter, channel
            'enable_retraining': True
        }

    def prune_model(self, model: 'CUDNT_Model', calibration_data: Optional[np.ndarray] = None) -> 'PrunedModel':
        """Prune model for reduced complexity."""
        logger.info(f"Pruning model with {self.config['sparsity']*100:.1f}% sparsity...")

        # Prune each layer
        pruned_parameters = {}
        pruning_masks = {}
        pruning_info = {}

        for layer_name, param_tensor in model.parameters.items():
            pruned_param, mask, prune_info = self._prune_parameter(
                param_tensor, layer_name, calibration_data
            )
            pruned_parameters[layer_name] = pruned_param
            pruning_masks[layer_name] = mask
            pruning_info[layer_name] = prune_info

        # Create pruned model
        pruned_model = PrunedModel(
            model.architecture,
            pruned_parameters,
            pruning_masks,
            pruning_info,
            self.config
        )

        self.pruning_stats = {
            'original_parameters': sum(np.prod(p.shape) for p in model.parameters.values()),
            'pruned_parameters': sum(np.sum(mask) for mask in pruning_masks.values()),
            'sparsity_achieved': 1 - (sum(np.sum(mask) for mask in pruning_masks.values()) /
                                    sum(np.prod(p.shape) for p in model.parameters.values())),
            'compression_ratio': self._calculate_pruning_compression(model.parameters, pruning_masks)
        }

        logger.info(".1f"        return pruned_model

    def _prune_parameter(self, param_tensor: np.ndarray, layer_name: str,
                        calibration_data: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Prune a single parameter tensor."""
        if self.config['method'] == 'magnitude':
            return self._magnitude_pruning(param_tensor, layer_name)
        elif self.config['method'] == 'structured':
            return self._structured_pruning(param_tensor, layer_name)
        elif self.config['method'] == 'l1':
            return self._l1_pruning(param_tensor, layer_name)
        else:
            raise ValueError(f"Unsupported pruning method: {self.config['method']}")

    def _magnitude_pruning(self, tensor: np.ndarray, layer_name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Magnitude-based pruning (unstructured)."""
        # Calculate threshold for target sparsity
        weights_flat = np.abs(tensor.flatten())
        threshold = np.percentile(weights_flat, self.config['sparsity'] * 100)

        # Create mask (1 for kept weights, 0 for pruned)
        mask = (np.abs(tensor) > threshold).astype(np.float32)

        # Apply pruning
        pruned_tensor = tensor * mask

        prune_info = {
            'method': 'magnitude',
            'threshold': float(threshold),
            'sparsity_target': self.config['sparsity'],
            'sparsity_actual': 1 - np.mean(mask),
            'weights_kept': int(np.sum(mask)),
            'weights_pruned': int(np.prod(tensor.shape) - np.sum(mask))
        }

        return pruned_tensor, mask, prune_info

    def _structured_pruning(self, tensor: np.ndarray, layer_name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Structured pruning (prune entire filters/channels)."""
        if len(tensor.shape) < 2:
            # Fallback to magnitude pruning for 1D tensors
            return self._magnitude_pruning(tensor, layer_name)

        # For convolutional layers, prune entire filters
        # For simplicity, prune based on L2 norm of filters
        if len(tensor.shape) == 4:  # Conv layer: [H, W, in_channels, out_channels]
            filter_norms = np.linalg.norm(tensor.reshape(tensor.shape[0], tensor.shape[1], -1), axis=(0, 1))
            filter_norms = filter_norms.reshape(tensor.shape[2], tensor.shape[3])

            # Prune along output channels
            channel_norms = np.linalg.norm(filter_norms, axis=0)
            n_keep = int(len(channel_norms) * (1 - self.config['sparsity']))
            keep_indices = np.argsort(channel_norms)[-n_keep:]

            mask = np.zeros_like(tensor)
            mask[:, :, :, keep_indices] = 1

        else:
            # Dense layer: prune along output dimension
            weight_norms = np.linalg.norm(tensor, axis=0)
            n_keep = int(len(weight_norms) * (1 - self.config['sparsity']))
            keep_indices = np.argsort(weight_norms)[-n_keep:]

            mask = np.zeros_like(tensor)
            mask[:, keep_indices] = 1

        pruned_tensor = tensor * mask

        prune_info = {
            'method': 'structured',
            'sparsity_target': self.config['sparsity'],
            'sparsity_actual': 1 - np.mean(mask),
            'structure': 'channel' if len(tensor.shape) == 4 else 'neuron'
        }

        return pruned_tensor, mask, prune_info

    def _l1_pruning(self, tensor: np.ndarray, layer_name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """L1 norm-based pruning."""
        # Similar to magnitude but using L1 norm for groups
        weights_flat = np.abs(tensor.flatten())
        threshold = np.percentile(weights_flat, self.config['sparsity'] * 100)
        mask = (np.abs(tensor) > threshold).astype(np.float32)
        pruned_tensor = tensor * mask

        prune_info = {
            'method': 'l1',
            'threshold': float(threshold),
            'sparsity_target': self.config['sparsity'],
            'sparsity_actual': 1 - np.mean(mask)
        }

        return pruned_tensor, mask, prune_info

    def _calculate_pruning_compression(self, original_params: Dict[str, np.ndarray],
                                     pruning_masks: Dict[str, np.ndarray]) -> float:
        """Calculate compression ratio from pruning."""
        original_params_total = sum(np.prod(p.shape) for p in original_params.values())
        remaining_params = sum(np.sum(mask) for mask in pruning_masks.values())
        return original_params_total / remaining_params if remaining_params > 0 else 1.0


class QuantizedModel:
    """Quantized model for efficient inference."""

    def __init__(self, architecture: List[Dict[str, Any]], parameters: Dict[str, np.ndarray],
                 quantization_info: Dict[str, Any], config: Dict[str, Any]):
        """Initialize quantized model."""
        self.architecture = architecture
        self.parameters = parameters
        self.quantization_info = quantization_info
        self.config = config

    def __call__(self, inputs):
        """Forward pass with quantization."""
        x = inputs
        for layer_config in self.architecture:
            layer_type = layer_config['type']
            layer_name = f"layer_{self.architecture.index(layer_config)}"

            if layer_type == 'dense':
                # Get quantized weights
                weight_key = f"{layer_name}_kernel"
                bias_key = f"{layer_name}_bias"

                if weight_key in self.parameters:
                    weights = self._dequantize_parameter(
                        self.parameters[weight_key],
                        self.quantization_info[weight_key]
                    )

                    # Apply linear transformation
                    x = np.matmul(x, weights)

                    # Add bias if available
                    if bias_key in self.parameters:
                        bias = self._dequantize_parameter(
                            self.parameters[bias_key],
                            self.quantization_info[bias_key]
                        )
                        x = x + bias

                # Apply activation
                if layer_config.get('activation') == 'relu':
                    x = np.maximum(x, 0)

        return x

    def _dequantize_parameter(self, quantized_tensor: np.ndarray, quant_info: Dict[str, Any]) -> np.ndarray:
        """Dequantize parameter for computation."""
        scale = quant_info['scale']
        zero_point = quant_info['zero_point']

        if quant_info['quantized_dtype'] == 'int8':
            return (quantized_tensor.astype(np.float32) - zero_point) * scale
        elif quant_info['quantized_dtype'] == 'int4':
            return quantized_tensor.astype(np.float32) * scale
        else:
            return quantized_tensor  # Already float32


class PrunedModel:
    """Pruned model with sparsity."""

    def __init__(self, architecture: List[Dict[str, Any]], parameters: Dict[str, np.ndarray],
                 pruning_masks: Dict[str, np.ndarray], pruning_info: Dict[str, Any],
                 config: Dict[str, Any]):
        """Initialize pruned model."""
        self.architecture = architecture
        self.parameters = parameters
        self.pruning_masks = pruning_masks
        self.pruning_info = pruning_info
        self.config = config

    def __call__(self, inputs):
        """Forward pass with pruning (masks already applied to weights)."""
        x = inputs
        for layer_config in self.architecture:
            layer_type = layer_config['type']

            if layer_type == 'dense':
                # Use pre-pruned weights
                layer_idx = self.architecture.index(layer_config)
                weight_key = f"layer_{layer_idx}_kernel"
                bias_key = f"layer_{layer_idx}_bias"

                if weight_key in self.parameters:
                    weights = self.parameters[weight_key]
                    x = np.matmul(x, weights)

                    if bias_key in self.parameters:
                        bias = self.parameters[bias_key]
                        x = x + bias

                if layer_config.get('activation') == 'relu':
                    x = np.maximum(x, 0)

        return x


class ModelCompressor:
    """
    Complete model compression pipeline.
    Combines quantization and pruning for optimal edge deployment.
    """

    def __init__(self, compression_config: Optional[Dict[str, Any]] = None):
        """Initialize model compressor."""
        self.config = compression_config or self._default_config()
        self.quantizer = ModelQuantizer(self.config.get('quantization', {}))
        self.pruner = ModelPruner(self.config.get('pruning', {}))

        logger.info("🗜️ Model Compressor initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default compression configuration."""
        return {
            'quantization': {
                'precision': 'int8',
                'per_channel': True,
                'symmetric': True
            },
            'pruning': {
                'method': 'magnitude',
                'sparsity': 0.3,
                'schedule': 'one_shot'
            },
            'target_size_mb': None,
            'max_accuracy_drop': 0.05
        }

    def compress_model(self, model: 'CUDNT_Model', calibration_data: Optional[np.ndarray] = None,
                      target_accuracy: Optional[float] = None) -> 'CompressedModel':
        """Compress model using quantization and pruning."""
        logger.info("Starting model compression...")

        # Step 1: Pruning
        logger.info("Step 1: Pruning...")
        pruned_model = self.pruner.prune_model(model, calibration_data)

        # Step 2: Quantization
        logger.info("Step 2: Quantization...")
        quantized_model = self.quantizer.quantize_model(pruned_model, calibration_data)

        # Step 3: Fine-tuning (optional)
        if self.config.get('enable_fine_tuning', False):
            logger.info("Step 3: Fine-tuning...")
            # Implement fine-tuning logic here
            pass

        # Create compressed model
        compressed_model = CompressedModel(
            model.architecture,
            quantized_model.parameters,
            quantized_model.quantization_info,
            pruned_model.pruning_masks,
            pruned_model.pruning_info,
            {
                'quantization_config': self.quantizer.config,
                'pruning_config': self.pruner.config,
                'compression_stats': {
                    'quantization_stats': self.quantizer.quantization_stats,
                    'pruning_stats': self.pruner.pruning_stats
                }
            }
        )

        logger.info("Model compression completed!")
        logger.info(".1f"        logger.info(".1f"
        return compressed_model

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            'quantization': self.quantizer.quantization_stats,
            'pruning': self.pruner.pruning_stats,
            'total_compression': (self.quantizer.quantization_stats.get('compression_ratio', 1) *
                                self.pruner.pruning_stats.get('compression_ratio', 1))
        }


class CompressedModel:
    """Fully compressed model for edge deployment."""

    def __init__(self, architecture: List[Dict[str, Any]], parameters: Dict[str, np.ndarray],
                 quantization_info: Dict[str, Any], pruning_masks: Dict[str, np.ndarray],
                 pruning_info: Dict[str, Any], compression_metadata: Dict[str, Any]):
        """Initialize compressed model."""
        self.architecture = architecture
        self.parameters = parameters
        self.quantization_info = quantization_info
        self.pruning_masks = pruning_masks
        self.pruning_info = pruning_info
        self.compression_metadata = compression_metadata

    def __call__(self, inputs):
        """Efficient inference with compressed model."""
        # Use quantized and pruned parameters for inference
        x = inputs
        for layer_config in self.architecture:
            layer_type = layer_config['type']
            layer_idx = self.architecture.index(layer_config)

            if layer_type == 'dense':
                weight_key = f"layer_{layer_idx}_kernel"
                bias_key = f"layer_{layer_idx}_bias"

                if weight_key in self.parameters:
                    # Dequantize weights for computation
                    weights = self._dequantize_parameter(
                        self.parameters[weight_key],
                        self.quantization_info[weight_key]
                    )

                    # Apply pruning mask
                    if weight_key in self.pruning_masks:
                        weights *= self.pruning_masks[weight_key]

                    x = np.matmul(x, weights)

                    if bias_key in self.parameters:
                        bias = self._dequantize_parameter(
                            self.parameters[bias_key],
                            self.quantization_info[bias_key]
                        )
                        x = x + bias

                if layer_config.get('activation') == 'relu':
                    x = np.maximum(x, 0)

        return x

    def _dequantize_parameter(self, quantized_tensor: np.ndarray, quant_info: Dict[str, Any]) -> np.ndarray:
        """Dequantize parameter for computation."""
        scale = quant_info['scale']
        zero_point = quant_info['zero_point']

        if quant_info['quantized_dtype'] == 'int8':
            return (quantized_tensor.astype(np.float32) - zero_point) * scale
        elif quant_info['quantized_dtype'] == 'int4':
            return quantized_tensor.astype(np.float32) * scale
        else:
            return quantized_tensor


# ===============================
# UTILITY FUNCTIONS
# ===============================

def create_model_quantizer(quantization_config: Optional[Dict[str, Any]] = None) -> ModelQuantizer:
    """Create model quantizer."""
    return ModelQuantizer(quantization_config)

def create_model_pruner(pruning_config: Optional[Dict[str, Any]] = None) -> ModelPruner:
    """Create model pruner."""
    return ModelPruner(pruning_config)

def create_model_compressor(compression_config: Optional[Dict[str, Any]] = None) -> ModelCompressor:
    """Create model compressor."""
    return ModelCompressor(compression_config)

def quick_compress_model(model: 'CUDNT_Model', compression_ratio: float = 0.5) -> CompressedModel:
    """Quick model compression with default settings."""
    config = {
        'quantization': {'precision': 'int8'},
        'pruning': {'sparsity': 1 - compression_ratio}
    }

    compressor = ModelCompressor(config)
    return compressor.compress_model(model)


# ===============================
# EXAMPLE USAGE
# ===============================

if __name__ == '__main__':
    print("🗜️ CUDNT Model Quantization & Pruning Demo")
    print("=" * 50)

    # Create a sample model
    from cudnt_production_system import create_cudnt_production
    cudnt = create_cudnt_production()

    architecture = [
        {'type': 'dense', 'units': 128, 'activation': 'relu'},
        {'type': 'dense', 'units': 64, 'activation': 'relu'},
        {'type': 'dense', 'units': 1}
    ]

    model = cudnt.create_model(architecture)
    print("✅ Sample model created")

    # Quantize model
    print("\nQuantizing model...")
    quantizer = create_model_quantizer({'precision': 'int8'})
    quantized_model = quantizer.quantize_model(model)
    print("✅ Model quantized")
    print(".1f"
    # Prune model
    print("\nPruning model...")
    pruner = create_model_pruner({'sparsity': 0.3, 'method': 'magnitude'})
    pruned_model = pruner.prune_model(model)
    print("✅ Model pruned")
    print(".1f"
    # Full compression
    print("\nFull compression...")
    compressor = create_model_compressor()
    compressed_model = compressor.compress_model(model)
    print("✅ Model fully compressed")

    compression_stats = compressor.get_compression_stats()
    print("
📊 Compression Results:"    print(".1f"    print(".1f"    print(".1f"
    print("\n✅ Model compression pipeline working!")
    print("🚀 Ready for edge deployment optimization")
