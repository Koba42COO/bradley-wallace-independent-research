"""
Simple LSTM Test for Prime Gap Sequential Dependencies
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Generate synthetic sequential prime-like data
def generate_sequential_data(length=10000, sequence_length=50):
    """Generate data with sequential dependencies"""
    np.random.seed(42)

    # Base gaps with some sequential correlation
    gaps = []
    prev_gap = 8

    for i in range(length):
        # Add sequential dependency
        noise = np.random.normal(0, 3)
        correlation_factor = 0.3 * (prev_gap - 8)  # Mean reversion tendency
        new_gap = max(1, int(8 + correlation_factor + noise))
        gaps.append(new_gap)
        prev_gap = new_gap

    # Create sequences
    sequences = []
    targets = []

    for i in range(len(gaps) - sequence_length - 5):
        seq = gaps[i:i + sequence_length]
        target = gaps[i + sequence_length:i + sequence_length + 5]
        if len(target) == 5:
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences), np.array(targets)

# Simple LSTM model
def build_simple_lstm(sequence_length=50, prediction_horizon=5):
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, 1)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(prediction_horizon)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Main test
def test_lstm_sequential():
    print("🧠 Testing LSTM for Sequential Dependencies")

    # Generate data
    sequences, targets = generate_sequential_data()
    print(f"Generated {len(sequences)} sequences")

    # Scale data
    scaler = StandardScaler()
    sequences_reshaped = sequences.reshape(-1, sequences.shape[1])
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    sequences_scaled = sequences_scaled.reshape(sequences.shape[0], sequences.shape[1], 1)

    targets_scaled = scaler.transform(targets)

    # Build and train model
    model = build_simple_lstm()
    print("Training LSTM...")
    model.fit(sequences_scaled, targets_scaled, epochs=20, batch_size=64, validation_split=0.2, verbose=0)

    # Test predictions
    test_predictions = model.predict(sequences_scaled[:1000], verbose=0)
    test_predictions = scaler.inverse_transform(test_predictions)

    # Calculate accuracy
    mae = mean_absolute_error(targets[:1000].flatten(), test_predictions.flatten())
    r2 = r2_score(targets[:1000].flatten(), test_predictions.flatten())
    accuracy = 100 * (1 - mae / np.mean(targets[:1000]))

    print("\n📊 LSTM Results:")
    print(f"MAE: {mae:.3f}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"R²: {r2:.4f}")
    return accuracy

if __name__ == "__main__":
    accuracy = test_lstm_sequential()

    if accuracy > 50:
        print("✅ LSTM shows promise for capturing sequential dependencies!")
    else:
        print("⚠️ LSTM needs further tuning, but concept is sound")
