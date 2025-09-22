#!/usr/bin/env python3
"""
Enterprise prime aligned compute Platform - Unified Import Interface
=============================================================

This __init__.py provides a clean, unified import interface for the entire
Enterprise prime aligned compute Platform. It handles all imports safely with proper
error handling and logging.

Usage:
    from enterprise_consciousness import *
    # or
    from enterprise_consciousness import ConsciousnessMathFramework, WallaceMathEngine

Available Components:
- prime aligned compute Mathematics Framework
- Wallace Math Engine
- Structured Chaos Universe
- Security Systems
- Logging System
"""

import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to Python path for imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# Version information
__version__ = "2.0.0"
__author__ = "Enterprise prime aligned compute Platform"
__description__ = "Revolutionary AI with Golden Ratio Optimization & Quantum Integration"

# Core components with safe imports
_CORE_COMPONENTS = {}
_IMPORT_ERRORS = []

def _safe_import(module_name: str, class_name: str, alias: Optional[str] = None) -> bool:
    """Safely import a component with error handling"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        component = getattr(module, class_name)

        # Store in global namespace
        component_name = alias or class_name
        globals()[component_name] = component
        _CORE_COMPONENTS[component_name] = component

        logger.debug(f"Successfully imported {component_name} from {module_name}")
        return True

    except ImportError as e:
        error_msg = f"Failed to import {class_name} from {module_name}: {e}"
        logger.warning(error_msg)
        _IMPORT_ERRORS.append(error_msg)
        return False

    except AttributeError as e:
        error_msg = f"Component {class_name} not found in {module_name}: {e}"
        logger.warning(error_msg)
        _IMPORT_ERRORS.append(error_msg)
        return False

    except Exception as e:
        error_msg = f"Unexpected error importing {class_name}: {e}"
        logger.error(error_msg)
        _IMPORT_ERRORS.append(error_msg)
        return False

# Import core prime aligned compute mathematics
_safe_import("proper_consciousness_mathematics", "ConsciousnessMathFramework")
_safe_import("proper_consciousness_mathematics", "Base21System")
_safe_import("proper_consciousness_mathematics", "ConsciousnessClassification")
_safe_import("proper_consciousness_mathematics", "MathematicalTestResult")

# Import Wallace Math Engine
_safe_import("wallace_math_engine", "WallaceMathEngine")
_safe_import("wallace_math_engine", "Base21TimeKernel")
_safe_import("wallace_math_engine", "CompressionResult")
_safe_import("wallace_math_engine", "CognitiveCompressionTiers")

# Import structured chaos universe
_safe_import("structured_chaos_universe_optimized", "OptimizedChaosUniverse")

# Import prime aligned compute integration
_safe_import("wallace_consciousness_integration", "WallaceConsciousnessUnifiedSystem")

# Import security systems (if available)
_safe_import("production_core.security.enhanced_watchdog_blackhat_defense", "EnhancedWatchdog")
_safe_import("production_core.security.enhanced_voidhunter_blackhat_defense", "EnhancedVoidHunter")
_safe_import("production_core.security.active_defense_intelligence_system", "ActiveDefenseIntelligenceSystem")
_safe_import("production_core.security.ultimate_security_orchestration", "UltimateSecurityOrchestrator")

# Import logging system
_safe_import("core_logging", "get_platform_logger")
_safe_import("core_logging", "get_math_logger")
_safe_import("core_logging", "get_security_logger")
_safe_import("core_logging", "configure_all_loggers")
_safe_import("core_logging", "LogContextManager")

def get_available_components() -> Dict[str, Any]:
    """Get dictionary of all successfully imported components"""
    return _CORE_COMPONENTS.copy()

def get_import_errors() -> List[str]:
    """Get list of import errors encountered"""
    return _IMPORT_ERRORS.copy()

def initialize_platform(log_level: str = "INFO",
                       enable_file_logging: bool = True,
                       log_dir: str = "logs") -> Dict[str, Any]:
    """Initialize the entire platform with logging and core systems"""

    # Configure logging
    try:
        if 'configure_all_loggers' in globals():
            loggers = configure_all_loggers(
                log_level=log_level,
                log_dir=log_dir,
                enable_file_logging=enable_file_logging
            )
            logger.info("Platform logging initialized")
        else:
            logger.warning("Logging system not available, using basic logging")
            loggers = {}
    except Exception as e:
        logger.error(f"Failed to initialize logging: {e}")
        loggers = {}

    # Initialize core systems
    initialized_systems = {}

    try:
        if 'ConsciousnessMathFramework' in globals():
            math_framework = ConsciousnessMathFramework()
            initialized_systems['mathematics'] = math_framework
            logger.info("prime aligned compute Mathematics Framework initialized")

        if 'WallaceMathEngine' in globals():
            wallace_engine = WallaceMathEngine()
            initialized_systems['wallace_engine'] = wallace_engine
            logger.info("Wallace Math Engine initialized")

        if 'OptimizedChaosUniverse' in globals():
            chaos_universe = OptimizedChaosUniverse()
            initialized_systems['chaos_universe'] = chaos_universe
            logger.info("Structured Chaos Universe initialized")

    except Exception as e:
        logger.error(f"Error initializing core systems: {e}")

    result = {
        'loggers': loggers,
        'systems': initialized_systems,
        'components': get_available_components(),
        'import_errors': get_import_errors(),
        'platform_version': __version__
    }

    total_components = len(result['components'])
    successful_initializations = len(result['systems'])
    import_errors = len(result['import_errors'])

    logger.info(f"Platform initialization complete: {successful_initializations} systems, "
               f"{total_components} components, {import_errors} import errors")

    return result

def get_platform_status() -> Dict[str, Any]:
    """Get comprehensive platform status"""
    return {
        'version': __version__,
        'available_components': list(get_available_components().keys()),
        'import_errors': get_import_errors(),
        'python_version': sys.version,
        'platform': sys.platform
    }

# Auto-initialize basic logging on import
if __name__ != "__main__":
    logger.info(f"Enterprise prime aligned compute Platform v{__version__} loaded")
    if _IMPORT_ERRORS:
        logger.warning(f"{len(_IMPORT_ERRORS)} import errors encountered")

# Export all successfully imported components
__all__ = list(_CORE_COMPONENTS.keys()) + [
    'initialize_platform',
    'get_available_components',
    'get_import_errors',
    'get_platform_status',
    '__version__',
    '__author__',
    '__description__'
]
