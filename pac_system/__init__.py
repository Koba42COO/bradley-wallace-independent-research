"""
PAC System package: unified access to PAC + Dual Kernel + Countercode

This package re-exports core entry points from the existing modules so we can
package and distribute without relocating the current files.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pac-system")
except PackageNotFoundError:  # fallback for dev
    __version__ = "0.0.0"

# Re-exports
from .unified import UnifiedConsciousnessSystem
from .validator import PACEntropyReversalValidator

__all__ = [
    "UnifiedConsciousnessSystem",
    "PACEntropyReversalValidator",
    "__version__",
]


