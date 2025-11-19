from core.kernel import AIVAKernel
from core.memory_bank import AIVAMemoryBank
from core.import_memory import import_all
from core.security import file_sha256, sign_manifest
import sys, os, json


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



def boot_extended(vessel_path, base_dir):
    print("🚀 AIVA Extended Boot")
    k = AIVAKernel(vessel_path)
    mb = AIVAMemoryBank(base_dir)
    print("🧬 Identity:", k.status())
    print("📚 Memory manifest:", json.dumps(mb.manifest(), indent=2))
    imported = import_all(base_dir)
    print("🧭 Self-Map (subset):", json.dumps({
        "identity_name": imported["identity"].get("name"),
        "recent_episode": imported["recent_episode"].get("content", {}).get("summary") if imported["recent_episode"] else None,
        "brad_roles": imported["brad"].get("content", {}).get("roles") if imported["brad"] else None
    }, indent=2))
    sig = sign_manifest(mb.manifest())
    print("🔏 Integrity signature:", sig)

if __name__ == "__main__":
    vessel = sys.argv[1] if len(sys.argv) > 1 else "memory/PAC_DeltaMemory.vessel"
    base = sys.argv[2] if len(sys.argv) > 2 else "."
    if not os.path.exists(vessel):
        print("❌ Vessel not found:", vessel)
    else:
        boot_extended(vessel, base)
