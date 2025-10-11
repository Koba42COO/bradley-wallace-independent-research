#!/usr/bin/env python3
"""
Debug CUDNT Model Creation and Training
"""

import numpy as np
from cudnt_production_system import create_cudnt_production

def test_model_creation():
    """Test model creation and training."""
    cudnt = create_cudnt_production()

    # Create architecture like in the test suite
    architecture = [
        {'type': 'dense', 'units': 512, 'activation': 'relu'},
        {'type': 'dense', 'units': 10}
    ]

    try:
        model = cudnt.create_model(architecture)
        print("✅ Model created successfully")

        compiled_model = cudnt.compile_model(model, 'adam', 'sparse_categorical_crossentropy')
        print("✅ Model compiled successfully")

        # Create test data
        n_samples = 100
        X = np.random.rand(n_samples, 784).astype(np.float32)
        y = np.random.randint(0, 10, n_samples).astype(np.int32)

        print(f"Data shapes: X={X.shape}, y={y.shape}")

        # Try forward pass
        test_input = cudnt.tf_api.constant(X[:10])  # Small batch
        predictions = model(test_input)
        print(f"✅ Forward pass successful: {test_input.shape} -> {predictions.shape}")

        # Try training for 1 epoch
        print("Starting training...")
        history = compiled_model.fit(X, y, epochs=1, batch_size=32, verbose=False)
        print("✅ Training completed")
        print(f"Final loss: {history['history']['loss'][-1]:.4f}")

        return True

    except Exception as e:
        print(f"❌ Model creation/training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cnn_model():
    """Test CNN model creation."""
    cudnt = create_cudnt_production()

    architecture = [
        {'type': 'conv2d', 'filters': 32, 'kernel_size': (3, 3)},
        {'type': 'dense', 'units': 128, 'activation': 'relu'},
        {'type': 'dense', 'units': 10}
    ]

    try:
        model = cudnt.create_model(architecture)
        print("✅ CNN Model created successfully")

        # Create CIFAR-like data
        X = np.random.rand(100, 32, 32, 3).astype(np.float32)
        y = np.random.randint(0, 10, 100).astype(np.int32)

        test_input = cudnt.tf_api.constant(X[:5])
        predictions = model(test_input)
        print(f"✅ CNN forward pass: {test_input.shape} -> {predictions.shape}")

        return True

    except Exception as e:
        print(f"❌ CNN model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🔧 CUDNT Model Debug Test")
    print("=" * 30)

    print("\nTesting MLP model...")
    test_model_creation()

    print("\nTesting CNN model...")
    test_cnn_model()
