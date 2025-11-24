import argparse, json, os, sys
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



# Ensure local package import works when run from root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.memory_bank import AIVAMemoryBank
from core.serializer import backup_memory
from core.rag import AIVARAG

ROOT = Path(__file__).resolve().parent

def cmd_show_identity(args):
    rag = AIVARAG(str(ROOT))
    ident = rag.who_am_i()
    print(json.dumps(ident, indent=2, ensure_ascii=False))

def cmd_manifest(args):
    mb = AIVAMemoryBank(str(ROOT))
    print(json.dumps(mb.manifest(), indent=2))

def cmd_add_episode(args):
    mem_path = ROOT / "data" / "memories" / "episodic.json"
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"episodes": []}
    if mem_path.exists():
        data = json.load(open(mem_path, "r", encoding="utf-8"))
    ep = {
        "id": args.id or f"manual_{len(data['episodes']):06d}",
        "time_utc": args.time or "<local>",
        "summary": args.summary,
        "participants": ["AIVA"],
        "outcomes": args.outcome or []
    }
    data["episodes"].append(ep)
    json.dump(data, open(mem_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("✓ Episode appended:", ep["id"])

def cmd_backup(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    backup_memory(str(ROOT), str(out))
    print("✓ Backup written:", out)

def main():
    ap = argparse.ArgumentParser(prog="aiva_cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("show"); s1_sub = s1.add_subparsers(dest="sub", required=True)
    s1i = s1_sub.add_parser("identity"); s1i.set_defaults(func=cmd_show_identity)

    s2 = sub.add_parser("manifest"); s2.set_defaults(func=cmd_manifest)

    s3 = sub.add_parser("add-episode")
    s3.add_argument("summary", help="Short description")
    s3.add_argument("--id", help="Episode ID", default=None)
    s3.add_argument("--time", help="ISO timestamp", default=None)
    s3.add_argument("--outcome", nargs="*", help="Outcome tags")
    s3.set_defaults(func=cmd_add_episode)

    s4 = sub.add_parser("backup")
    s4.add_argument("out", help="Output .tgz path")
    s4.set_defaults(func=cmd_backup)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
