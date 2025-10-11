import argparse
import json
import numpy as np

from pac_system.unified import UnifiedConsciousnessSystem
from pac_system.validator import PACEntropyReversalValidator


def cmd_validate_entropy(args: argparse.Namespace) -> int:
    data = np.random.randn(args.rows, args.cols)
    validator = PACEntropyReversalValidator(prime_scale=args.prime_scale)
    results = validator.validate_entropy_reversal(data, n_experiments=args.n)
    print(json.dumps(results, default=str, indent=2))
    return 0


def cmd_run_unified(args: argparse.Namespace) -> int:
    system = UnifiedConsciousnessSystem(prime_scale=args.prime_scale,
                                        consciousness_weight=args.weight)
    if args.mode == "entropy":
        x = np.random.randn(10, 5)
    elif args.mode == "ai":
        x = np.random.randn(100, 20)
    elif args.mode == "memory":
        x = "Consciousness mathematics creates order from chaos"
    elif args.mode == "storage":
        x = {"key": "value", "phi": 1.618034}
    else:
        x = np.random.randn(50, 10)

    result = system.process_universal_optimization(x, optimization_type=args.mode)
    print(json.dumps(result, default=str)[:2000])
    return 0


def cmd_gen_datasets(args: argparse.Namespace) -> int:
    from clean_dataset_generator import CleanDatasetGenerator

    scales = []
    for token in args.scales.split(","):
        token = token.strip()
        if token.endswith("e6"):
            scales.append(10**6)
        elif token.endswith("e7"):
            scales.append(10**7)
        elif token.endswith("e8"):
            scales.append(10**8)
        elif token.endswith("e9"):
            scales.append(10**9)
        else:
            try:
                scales.append(int(token))
            except Exception:
                pass

    gen = CleanDatasetGenerator()
    gen.scales = scales or gen.scales
    datasets = gen.generate_all_datasets()
    print(json.dumps({k: v.__dict__ for k, v in datasets.items()}, default=str, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="pacctl", description="PAC System CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate-entropy")
    p_val.add_argument("--rows", type=int, default=100)
    p_val.add_argument("--cols", type=int, default=10)
    p_val.add_argument("-n", type=int, default=20)
    p_val.add_argument("--prime-scale", type=int, default=100000)
    p_val.set_defaults(func=cmd_validate_entropy)

    p_run = sub.add_parser("run-unified")
    p_run.add_argument("--mode", default="auto",
                       choices=["auto", "entropy", "ai", "memory", "storage"])
    p_run.add_argument("--prime-scale", type=int, default=50000)
    p_run.add_argument("--weight", type=float, default=0.79)
    p_run.set_defaults(func=cmd_run_unified)

    p_gen = sub.add_parser("gen-datasets")
    p_gen.add_argument("--scales", default="1e6,1e7,1e8")
    p_gen.set_defaults(func=cmd_gen_datasets)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


