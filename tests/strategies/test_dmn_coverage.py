#!/usr/bin/env python3
"""Test script to verify DMN LSTM Strategy methods work correctly"""

import numpy as np
import pandas as pd
from unittest.mock import Mock
from datetime import datetime

from qframe.strategies.research.dmn_lstm_strategy import DMNLSTMStrategy, DMNConfig

def create_test_data():
    """Create test market data"""
    dates = pd.date_range('2023-01-01', periods=200, freq='1h')
    np.random.seed(42)

    data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(200) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(200) * 0.1) + abs(np.random.randn(200) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(200) * 0.1) - abs(np.random.randn(200) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(200) * 0.1),
        'volume': abs(np.random.randn(200) * 1000 + 5000)
    })
    data.set_index('timestamp', inplace=True)
    return data

def test_new_methods():
    """Test all newly added methods"""
    print("Testing DMN LSTM Strategy new methods...")

    # Create strategy
    config = DMNConfig(
        window_size=10,
        hidden_size=8,
        num_layers=1,
        epochs=2,
        batch_size=4,
        learning_rate=0.01
    )

    strategy = DMNLSTMStrategy(
        config=config,
        feature_processor=Mock(),
        metrics_collector=Mock()
    )

    # Create test data
    data = create_test_data()

    print("\n1. Testing train method...")
    try:
        # Train the model
        history = strategy.train(data[:100])
        print(f"✓ Training completed. Losses: {history.get('train_losses', [])[:3]}...")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False

    print("\n2. Testing predict method...")
    try:
        predictions = strategy.predict(data[100:150])
        print(f"✓ Predictions generated. Shape: {predictions.shape}, Sample: {predictions[:3]}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

    print("\n3. Testing evaluate method...")
    try:
        eval_results = strategy.evaluate(data[100:200])
        metrics = eval_results.get('metrics', {})
        print(f"✓ Evaluation completed. Directional accuracy: {metrics.get('directional_accuracy', 0):.2%}")
        print(f"  Signals generated: {eval_results['signals']['total']}")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False

    print("\n4. Testing save_model method...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            strategy.save_model(tmp.name)
            print(f"✓ Model saved to {tmp.name}")

            # Test load_model
            print("\n5. Testing load_model method...")
            new_strategy = DMNLSTMStrategy(
                config=config,
                feature_processor=Mock(),
                metrics_collector=Mock()
            )
            new_strategy.load_model(tmp.name)
            print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Save/Load failed: {e}")
        return False

    print("\n6. Testing _predictions_to_signals...")
    try:
        from qframe.strategies.research.dmn_lstm_strategy import SignalAction
        predictions = np.array([0.8, -0.6, 0.2, -0.3, 0.9])
        timestamps = data.index[:5]
        prices = data['close'].iloc[:5]

        signals = strategy._predictions_to_signals(predictions, timestamps, prices)
        print(f"✓ Generated {len(signals)} signals")
        for s in signals[:2]:
            print(f"  - {s.action.value}: confidence={s.confidence.value}, strength={s.strength}")
    except Exception as e:
        print(f"✗ Signal generation failed: {e}")
        return False

    print("\n7. Testing get_strategy_info...")
    try:
        info = strategy.get_strategy_info()
        print(f"✓ Strategy info: {info['name']}, trained={info['training_status']['is_trained']}")
    except Exception as e:
        print(f"✗ Get info failed: {e}")
        return False

    print("\n✅ All DMN LSTM Strategy methods tested successfully!")
    return True

if __name__ == "__main__":
    success = test_new_methods()
    exit(0 if success else 1)