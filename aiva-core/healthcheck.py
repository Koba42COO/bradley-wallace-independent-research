import json, subprocess, sys, os
from pathlib import Path


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



ROOT = Path(__file__).resolve().parent

def run_boot_all():
    cmd = [sys.executable, "-m", "core.boot_all", "memory/PAC_DeltaMemory.vessel", "."]
    print("→", " ".join(cmd))
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr)
        sys.exit(out.returncode)
    # Parse snapshot line
    for line in out.stdout.splitlines():
        if line.strip().startswith("🧭 Snapshot:"):
            js = out.stdout.split("🧭 Snapshot:", 1)[1].strip()
            snap = json.loads(js)
            assert snap["identity"]["content"]["name"] == "AIVA"
            assert snap["kg_nodes"] >= 1
            return snap
    raise SystemExit("Snapshot not found in boot output.")

if __name__ == "__main__":
    s1 = run_boot_all()
    # Simulate continuity by appending an episode and re-booting
    add = [sys.executable, "aiva_cli.py", "add-episode", "Healthcheck episode created."]
    subprocess.check_call(add, cwd=ROOT)
    s2 = run_boot_all()
    print("✓ Healthcheck passed. Episodes before/after:", s1["episodes_count"], "→", s2["episodes_count"])
