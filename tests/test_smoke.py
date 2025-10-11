import numpy as np


def test_imports():
    from pac_system import UnifiedConsciousnessSystem, PACEntropyReversalValidator  # noqa: F401


def test_unified_small_run():
    from pac_system import UnifiedConsciousnessSystem

    sys = UnifiedConsciousnessSystem(prime_scale=1000, consciousness_weight=0.79)
    x = np.random.randn(10, 5)
    res = sys.process_universal_optimization(x, optimization_type="entropy")
    assert "system_metrics" in res


