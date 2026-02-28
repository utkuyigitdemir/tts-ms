"""
Automatic Engine Setup and Dependency Management.

This module provides automatic detection and installation of engine-specific
dependencies at startup. It checks Python version compatibility, pip packages,
and system dependencies for each TTS engine.

Features:
    - Engine requirements registry with all dependency information
    - Python version compatibility checking
    - Pip package installation (optional auto-install)
    - System dependency detection (ffmpeg, espeak-ng)
    - Clear error messages with installation instructions

Usage:
    from tts_ms.core.engine_setup import setup_engine, check_requirements

    # Check requirements without installing
    result = check_requirements("chatterbox")
    if not result.satisfied:
        print(result.message)

    # Auto-setup engine (install missing packages)
    setup_engine("f5tts", auto_install=True)

Environment Variables:
    TTS_MS_AUTO_INSTALL=1  - Enable automatic pip package installation
    TTS_MS_SKIP_SETUP=1    - Skip all setup checks (for testing)
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class SetupStatus(Enum):
    """Status of engine setup check."""
    OK = "ok"
    PYTHON_VERSION_ERROR = "python_version_error"
    MISSING_PACKAGES = "missing_packages"
    MISSING_SYSTEM_DEPS = "missing_system_deps"
    INSTALL_FAILED = "install_failed"


@dataclass
class EngineRequirements:
    """
    Requirements specification for a TTS engine.

    Attributes:
        name: Engine identifier (e.g., "piper", "f5tts")
        display_name: Human-readable engine name
        python_min: Minimum Python version (e.g., (3, 10))
        python_max: Maximum Python version (e.g., (3, 11)) or None for no max
        pip_packages: List of pip packages to install
        pip_extras: Extra pip packages for optional features
        system_deps: System dependencies (ffmpeg, espeak-ng, etc.)
        env_name: Recommended conda environment name
        notes: Additional setup notes
        check_import: Module to import to verify installation
        pre_install_hook: Optional function to run before pip install
        post_install_hook: Optional function to run after pip install
    """
    name: str
    display_name: str
    python_min: Tuple[int, int] = (3, 10)
    python_max: Optional[Tuple[int, int]] = None
    pip_packages: List[str] = field(default_factory=list)
    pip_extras: List[str] = field(default_factory=list)
    system_deps: List[str] = field(default_factory=list)
    env_name: str = "tts"
    notes: str = ""
    check_import: Optional[str] = None
    pre_install_hook: Optional[Callable[[], bool]] = None
    post_install_hook: Optional[Callable[[], bool]] = None


@dataclass
class SetupResult:
    """Result of setup/check operation."""
    status: SetupStatus
    satisfied: bool
    message: str
    missing_packages: List[str] = field(default_factory=list)
    missing_system_deps: List[str] = field(default_factory=list)
    installed_packages: List[str] = field(default_factory=list)


# =============================================================================
# Engine Requirements Registry
# =============================================================================

ENGINE_REGISTRY: Dict[str, EngineRequirements] = {
    "piper": EngineRequirements(
        name="piper",
        display_name="Piper TTS",
        python_min=(3, 10),
        python_max=None,  # Works with any Python 3.10+
        pip_packages=["piper-tts"],
        system_deps=[],
        env_name="tts",
        check_import="piper",
        notes="CPU-only engine, no GPU required. Fastest option.",
    ),

    "f5tts": EngineRequirements(
        name="f5tts",
        display_name="F5-TTS",
        python_min=(3, 10),
        python_max=None,
        pip_packages=[
            "torch==2.5.0",
            "torchaudio==2.5.0",
            "f5-tts",
        ],
        pip_extras=["--index-url", "https://download.pytorch.org/whl/cpu"],
        system_deps=["ffmpeg"],
        env_name="tts",
        check_import="f5_tts",
        notes="Voice cloning model. Requires ffmpeg for audio processing. "
              "Use torchaudio 2.5.0 on Windows to avoid DLL issues.",
    ),

    "styletts2": EngineRequirements(
        name="styletts2",
        display_name="StyleTTS2",
        python_min=(3, 10),
        python_max=None,
        pip_packages=[
            "torch==2.5.0",
            "torchaudio==2.5.0",
            "styletts2",
        ],
        pip_extras=["--index-url", "https://download.pytorch.org/whl/cpu"],
        system_deps=["espeak-ng"],
        env_name="tts",
        check_import="styletts2",
        notes="High-quality TTS with style transfer. Requires espeak-ng for phonemization. "
              "Run: python -c \"import nltk; nltk.download('punkt_tab')\" after install.",
    ),

    "legacy": EngineRequirements(
        name="legacy",
        display_name="Legacy XTTS v2 (Coqui)",
        python_min=(3, 10),
        python_max=(3, 11),  # Coqui TTS doesn't support Python 3.12+
        pip_packages=["coqui-tts"],
        system_deps=[],
        env_name="tts",
        check_import="TTS",
        notes="Coqui TTS fork. Python 3.10-3.11 only. Large model download (~2GB) on first run.",
    ),

    "chatterbox": EngineRequirements(
        name="chatterbox",
        display_name="Chatterbox (ResembleAI)",
        python_min=(3, 10),
        python_max=(3, 11),  # numpy<1.26 incompatible with Python 3.12
        pip_packages=[
            "torch==2.6.0",
            "torchaudio==2.6.0",
            "chatterbox-tts",
        ],
        pip_extras=["--index-url", "https://download.pytorch.org/whl/cpu"],
        system_deps=[],
        env_name="tts311",
        check_import="chatterbox",
        notes="Requires Python 3.11 due to numpy<1.26 constraint. "
              "Multilingual support with voice cloning.",
    ),

    "cosyvoice": EngineRequirements(
        name="cosyvoice",
        display_name="CosyVoice (Alibaba)",
        python_min=(3, 10),
        python_max=(3, 10),  # CosyVoice specifically requires 3.10
        pip_packages=[],  # No pip package - manual git install required
        system_deps=[],  # sox is Linux only, skip on Windows
        env_name="tts310",
        check_import="cosyvoice",
        notes="Requires Python 3.10 and manual installation from git: "
              "git clone https://github.com/FunAudioLLM/CosyVoice && "
              "cd CosyVoice && pip install -r requirements.txt",
    ),

    "kokoro": EngineRequirements(
        name="kokoro",
        display_name="Kokoro TTS (ONNX)",
        python_min=(3, 10),
        python_max=None,
        pip_packages=["kokoro-onnx"],
        system_deps=[],
        env_name="tts",
        check_import="kokoro_onnx",
        notes="CPU-only ONNX engine. Download model files from "
              "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX",
    ),

    "qwen3tts": EngineRequirements(
        name="qwen3tts",
        display_name="Qwen3-TTS (Alibaba)",
        python_min=(3, 10),
        python_max=None,
        pip_packages=[
            "torch",
            "torchaudio",
            "qwen-tts",
        ],
        system_deps=[],
        env_name="tts",
        check_import="qwen_tts",
        notes="GPU recommended (~3-4 GB VRAM). Supports preset speakers "
              "and voice cloning.",
    ),

    "vibevoice": EngineRequirements(
        name="vibevoice",
        display_name="VibeVoice (Microsoft)",
        python_min=(3, 10),
        python_max=None,
        pip_packages=["vibevoice"],
        system_deps=[],
        env_name="tts",
        check_import="vibevoice",
        notes="Research-only license. GPU required (~7 GB VRAM for 1.5B model). "
              "ffmpeg recommended if loading non-WAV reference audio.",
    ),
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_python_version() -> Tuple[int, int]:
    """Get current Python version as (major, minor) tuple."""
    return (sys.version_info.major, sys.version_info.minor)


def check_python_version(
    python_min: Tuple[int, int],
    python_max: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, str]:
    """
    Check if current Python version is compatible.

    Args:
        python_min: Minimum required version (major, minor)
        python_max: Maximum allowed version (major, minor) or None

    Returns:
        Tuple of (is_compatible, message)
    """
    current = get_python_version()
    current_str = f"{current[0]}.{current[1]}"

    if current < python_min:
        min_str = f"{python_min[0]}.{python_min[1]}"
        return False, f"Python {current_str} is below minimum {min_str}"

    if python_max and current > python_max:
        max_str = f"{python_max[0]}.{python_max[1]}"
        return False, f"Python {current_str} exceeds maximum {max_str}"

    return True, f"Python {current_str} is compatible"


def is_package_installed(package_name: str) -> bool:
    """
    Check if a pip package is installed.

    Handles version specifiers (e.g., "torch==2.5.0" -> checks for "torch").
    """
    # Extract base package name (remove version specifier)
    base_name = package_name.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0]
    base_name = base_name.replace("-", "_").replace(".", "_")

    # Special cases for package name vs import name mismatches
    import_map = {
        "piper_tts": "piper",
        "f5_tts": "f5_tts",
        "styletts2": "styletts2",
        "coqui_tts": "TTS",
        "chatterbox_tts": "chatterbox",
        "torch": "torch",
        "torchaudio": "torchaudio",
        "kokoro_onnx": "kokoro_onnx",
        "qwen_tts": "qwen_tts",
        "vibevoice": "vibevoice",
    }

    import_name = import_map.get(base_name, base_name)

    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def is_system_dep_available(dep: str) -> bool:
    """Check if a system dependency is available in PATH."""
    return shutil.which(dep) is not None


def get_missing_packages(packages: List[str]) -> List[str]:
    """Get list of packages that are not installed."""
    return [pkg for pkg in packages if not is_package_installed(pkg)]


def get_missing_system_deps(deps: List[str]) -> List[str]:
    """Get list of system dependencies that are not available."""
    return [dep for dep in deps if not is_system_dep_available(dep)]


def install_packages(
    packages: List[str],
    extras: Optional[List[str]] = None,
    quiet: bool = False,
) -> Tuple[bool, str]:
    """
    Install pip packages.

    Args:
        packages: List of packages to install
        extras: Extra pip arguments (e.g., --index-url)
        quiet: Suppress pip output

    Returns:
        Tuple of (success, message)
    """
    if not packages:
        return True, "No packages to install"

    cmd = [sys.executable, "-m", "pip", "install"]

    if quiet:
        cmd.append("-q")

    # Add extras first (like --index-url)
    if extras:
        cmd.extend(extras)

    cmd.extend(packages)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode == 0:
            return True, f"Successfully installed: {', '.join(packages)}"
        else:
            return False, f"pip install failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Installation timed out after 10 minutes"
    except Exception as e:
        return False, f"Installation error: {e}"


# =============================================================================
# Main API Functions
# =============================================================================

def check_requirements(engine_type: str) -> SetupResult:
    """
    Check if all requirements are satisfied for an engine.

    Args:
        engine_type: Engine identifier (e.g., "piper", "f5tts")

    Returns:
        SetupResult with status and details
    """
    if engine_type not in ENGINE_REGISTRY:
        return SetupResult(
            status=SetupStatus.OK,
            satisfied=True,
            message=f"Unknown engine '{engine_type}' - skipping requirements check",
        )

    reqs = ENGINE_REGISTRY[engine_type]

    # Check Python version
    py_ok, py_msg = check_python_version(reqs.python_min, reqs.python_max)
    if not py_ok:
        env_hint = f" Use conda environment '{reqs.env_name}' instead." if reqs.env_name != "tts" else ""
        return SetupResult(
            status=SetupStatus.PYTHON_VERSION_ERROR,
            satisfied=False,
            message=f"{reqs.display_name}: {py_msg}.{env_hint}",
        )

    # Check pip packages
    missing_pkgs = get_missing_packages(reqs.pip_packages)

    # Check system dependencies
    missing_deps = get_missing_system_deps(reqs.system_deps)

    # Build result
    if missing_pkgs or missing_deps:
        messages = []

        if missing_pkgs:
            messages.append(f"Missing packages: {', '.join(missing_pkgs)}")

        if missing_deps:
            messages.append(f"Missing system deps: {', '.join(missing_deps)}")

        if reqs.notes:
            messages.append(f"Note: {reqs.notes}")

        return SetupResult(
            status=SetupStatus.MISSING_PACKAGES if missing_pkgs else SetupStatus.MISSING_SYSTEM_DEPS,
            satisfied=False,
            message=f"{reqs.display_name}: " + " | ".join(messages),
            missing_packages=missing_pkgs,
            missing_system_deps=missing_deps,
        )

    return SetupResult(
        status=SetupStatus.OK,
        satisfied=True,
        message=f"{reqs.display_name}: All requirements satisfied",
    )


def setup_engine(
    engine_type: str,
    auto_install: bool = False,
    quiet: bool = False,
) -> SetupResult:
    """
    Setup an engine by checking and optionally installing dependencies.

    Args:
        engine_type: Engine identifier
        auto_install: If True, automatically install missing pip packages
        quiet: Suppress installation output

    Returns:
        SetupResult with setup status
    """
    # Check if setup is disabled
    if os.getenv("TTS_MS_SKIP_SETUP") == "1":
        return SetupResult(
            status=SetupStatus.OK,
            satisfied=True,
            message="Setup check skipped (TTS_MS_SKIP_SETUP=1)",
        )

    # Get requirements
    if engine_type not in ENGINE_REGISTRY:
        return SetupResult(
            status=SetupStatus.OK,
            satisfied=True,
            message=f"Unknown engine '{engine_type}' - no setup required",
        )

    reqs = ENGINE_REGISTRY[engine_type]

    # Check Python version first
    py_ok, py_msg = check_python_version(reqs.python_min, reqs.python_max)
    if not py_ok:
        env_hint = ""
        if reqs.env_name != "tts":
            env_hint = "\n\nTo fix, create the correct environment:\n"
            env_hint += f"  conda create -n {reqs.env_name} python={reqs.python_min[0]}.{reqs.python_min[1]} -y\n"
            env_hint += f"  conda activate {reqs.env_name}\n"
            env_hint += "  pip install -e /path/to/tts-ms"

        return SetupResult(
            status=SetupStatus.PYTHON_VERSION_ERROR,
            satisfied=False,
            message=f"{reqs.display_name}: {py_msg}{env_hint}",
        )

    # Check and optionally install pip packages
    missing_pkgs = get_missing_packages(reqs.pip_packages)
    installed = []

    if missing_pkgs:
        if auto_install or os.getenv("TTS_MS_AUTO_INSTALL") == "1":
            # Run pre-install hook if defined
            if reqs.pre_install_hook:
                reqs.pre_install_hook()

            # Install missing packages
            success, msg = install_packages(missing_pkgs, reqs.pip_extras, quiet)

            if not success:
                return SetupResult(
                    status=SetupStatus.INSTALL_FAILED,
                    satisfied=False,
                    message=f"{reqs.display_name}: {msg}",
                    missing_packages=missing_pkgs,
                )

            installed = missing_pkgs

            # Run post-install hook if defined
            if reqs.post_install_hook:
                reqs.post_install_hook()

            # Re-check packages
            still_missing = get_missing_packages(reqs.pip_packages)
            if still_missing:
                return SetupResult(
                    status=SetupStatus.INSTALL_FAILED,
                    satisfied=False,
                    message=f"{reqs.display_name}: Installation completed but packages still missing: {still_missing}",
                    missing_packages=still_missing,
                )
        else:
            # No auto-install - return instructions
            install_cmd = f"pip install {' '.join(missing_pkgs)}"
            if reqs.pip_extras:
                install_cmd = f"pip install {' '.join(reqs.pip_extras)} {' '.join(missing_pkgs)}"

            return SetupResult(
                status=SetupStatus.MISSING_PACKAGES,
                satisfied=False,
                message=f"{reqs.display_name}: Missing packages. Run:\n  {install_cmd}",
                missing_packages=missing_pkgs,
            )

    # Check system dependencies (can't auto-install)
    missing_deps = get_missing_system_deps(reqs.system_deps)
    if missing_deps:
        hints = []
        for dep in missing_deps:
            if dep == "ffmpeg":
                hints.append("ffmpeg: pip install imageio-ffmpeg (or apt/brew install ffmpeg)")
            elif dep == "espeak-ng":
                hints.append("espeak-ng: apt install espeak-ng (or brew install espeak-ng)")
            elif dep == "sox":
                hints.append("sox: apt install sox libsox-dev (Linux only)")
            else:
                hints.append(f"{dep}: Install via system package manager")

        return SetupResult(
            status=SetupStatus.MISSING_SYSTEM_DEPS,
            satisfied=False,
            message=f"{reqs.display_name}: Missing system dependencies:\n  " + "\n  ".join(hints),
            missing_system_deps=missing_deps,
        )

    # All checks passed
    msg = f"{reqs.display_name}: Ready"
    if installed:
        msg += f" (installed: {', '.join(installed)})"

    return SetupResult(
        status=SetupStatus.OK,
        satisfied=True,
        message=msg,
        installed_packages=installed,
    )


def get_engine_info(engine_type: str) -> Optional[EngineRequirements]:
    """Get requirements info for an engine."""
    return ENGINE_REGISTRY.get(engine_type)


def list_engines() -> Dict[str, EngineRequirements]:
    """Get all registered engines."""
    return ENGINE_REGISTRY.copy()


def print_engine_status(engine_type: Optional[str] = None) -> None:
    """
    Print status of engine(s) to console.

    Args:
        engine_type: Specific engine or None for all engines
    """
    engines = [engine_type] if engine_type else list(ENGINE_REGISTRY.keys())
    current_py = get_python_version()

    print(f"\nEngine Status (Python {current_py[0]}.{current_py[1]})")
    print("=" * 60)

    for eng in engines:
        if eng not in ENGINE_REGISTRY:
            print(f"  {eng}: Unknown engine")
            continue

        result = check_requirements(eng)
        reqs = ENGINE_REGISTRY[eng]

        status_icon = "OK" if result.satisfied else "!!"
        py_range = f"{reqs.python_min[0]}.{reqs.python_min[1]}"
        if reqs.python_max:
            py_range += f"-{reqs.python_max[0]}.{reqs.python_max[1]}"
        else:
            py_range += "+"

        print(f"\n  [{status_icon}] {reqs.display_name} ({eng})")
        print(f"      Python: {py_range} | Env: {reqs.env_name}")

        if result.missing_packages:
            print(f"      Missing: {', '.join(result.missing_packages)}")

        if result.missing_system_deps:
            print(f"      System deps: {', '.join(result.missing_system_deps)}")

        if reqs.notes and not result.satisfied:
            print(f"      Note: {reqs.notes[:60]}...")

    print()


# =============================================================================
# Integration Functions
# =============================================================================

def ensure_engine_ready(engine_type: str, auto_install: bool = False) -> None:
    """
    Ensure an engine is ready to use, raising an error if not.

    This is called by the engine factory before creating an engine instance.
    Skipped when TTS_MS_SKIP_SETUP=1 is set (useful for testing/CI).

    Args:
        engine_type: Engine identifier
        auto_install: Enable automatic pip package installation

    Raises:
        RuntimeError: If engine requirements are not satisfied
    """
    if os.getenv("TTS_MS_SKIP_SETUP") == "1":
        return

    result = setup_engine(engine_type, auto_install=auto_install)

    if not result.satisfied:
        raise RuntimeError(
            f"Engine '{engine_type}' requirements not satisfied.\n\n{result.message}"
        )
