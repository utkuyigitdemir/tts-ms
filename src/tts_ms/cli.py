"""
Command-Line Interface for tts-ms.

This module provides serverless TTS synthesis without running the HTTP server.
It supports single text synthesis, batch processing from files, and dry-run
mode for testing without actual synthesis.

Usage Examples:
    # Single text synthesis
    tts-ms --text "Merhaba dünya" --out hello.wav

    # Positional text (same as above)
    tts-ms "Merhaba dünya" --out hello.wav

    # Batch processing from file
    tts-ms --file inputs.txt --out output_dir/

    # Dry-run mode (no synthesis, shows chunking info)
    tts-ms --text "Test" --dry-run --json

    # Override speaker and language
    tts-ms --text "Test" --speaker female --language tr

    # Check engine status and requirements
    tts-ms --engines

    # Setup specific engine (with auto-install)
    tts-ms --setup f5tts

Environment Variables:
    TTS_MODEL_TYPE: Engine to use (piper, f5tts, legacy, etc.)
    TTS_DEVICE: Device override (cuda/cpu)
    TTS_MS_SPEAKER: Default speaker
    TTS_MS_LANGUAGE: Default language
    TTS_MS_AUTO_INSTALL: Enable automatic pip package installation
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from tts_ms.core.config import load_settings
from tts_ms.core.logging import configure_logging, get_logger, info, set_request_id
from tts_ms.tts.chunker import chunk_text
from tts_ms.tts.storage import hash_dict, make_key
from tts_ms.utils.text import normalize_tr


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv: Optional list of arguments (defaults to sys.argv).

    Returns:
        Parsed argument namespace with all CLI options.
    """
    parser = argparse.ArgumentParser(description="tts-ms CLI (serverless synth)")

    # Input options (mutually exclusive: text vs file)
    parser.add_argument("text_pos", nargs="?", help="Text to synthesize (positional)")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--file", help="Batch input file (1 line = 1 item)")

    # Output options
    parser.add_argument("--out", help="Output path (file or dir in batch mode)")

    # TTS configuration overrides
    parser.add_argument("--speaker", help="Speaker override")
    parser.add_argument("--language", help="Language override")
    parser.add_argument("--device", help="Device override (cuda/cpu)")

    # Execution modes
    parser.add_argument("--dry-run", action="store_true",
                       help="Parse and summarize without synth")
    parser.add_argument("--json", action="store_true",
                       help="Print JSON summary")

    # Engine management
    parser.add_argument("--engines", action="store_true",
                       help="Show all engines and their status")
    parser.add_argument("--setup", metavar="ENGINE",
                       help="Setup specific engine (check/install dependencies)")
    parser.add_argument("--auto-install", action="store_true",
                       help="Auto-install missing pip packages (use with --setup)")

    return parser.parse_args(argv)


def _load_texts(args: argparse.Namespace) -> List[str]:
    """
    Load input texts from arguments or file.

    Handles three input modes:
        1. Positional text argument
        2. --text flag
        3. --file for batch processing (one text per line)

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of texts to synthesize.

    Raises:
        SystemExit: If no input provided or conflicting options used.
    """
    text = args.text or args.text_pos

    # Batch mode: read texts from file
    if args.file:
        if text:
            raise SystemExit("Use --file without --text or positional text.")
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        items = [line.strip() for line in lines if line.strip()]
        if not items:
            raise SystemExit("Input file is empty.")
        return items

    # Single text mode
    if not text:
        raise SystemExit("Provide --text or a positional text.")
    return [text]


def _resolve_output_paths(args: argparse.Namespace, count: int) -> List[Path]:
    """
    Determine output file paths based on mode and arguments.

    For batch mode (--file): Creates numbered files in output directory.
    For single mode: Uses specified path or defaults to 'out.wav'.

    Args:
        args: Parsed command-line arguments.
        count: Number of output files needed.

    Returns:
        List of Path objects for output WAV files.
    """
    # Batch mode: create output directory with numbered files
    if args.file:
        out_dir = Path(args.out or "out")
        out_dir.mkdir(parents=True, exist_ok=True)
        return [out_dir / f"item_{i + 1:03d}.wav" for i in range(count)]

    # Single mode: use specified path or default
    out_path = Path(args.out or "out.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return [out_path]


def _summary_for_text(
    text: str,
    speaker: str,
    language: str,
    device: str,
    engine_type: str,
    model_id: str,
    settings_hash: str,
) -> dict:
    """
    Generate a dry-run summary for a text without synthesis.

    This function normalizes the text, chunks it, and generates cache keys
    to show what would happen during actual synthesis.

    Args:
        text: Input text to analyze.
        speaker: Selected speaker voice.
        language: Target language code.
        device: Compute device (cuda/cpu).
        engine_type: TTS engine name.
        model_id: Model identifier.
        settings_hash: Hash of engine settings for cache keying.

    Returns:
        Dictionary with analysis results (text_len, chunks, keys, etc.).
    """
    # Normalize Turkish text (i/İ handling, punctuation)
    norm, _ = normalize_tr(text)

    # Chunk text for synthesis (respecting sentence boundaries)
    cr = chunk_text(norm, max_chars=220)

    # Generate cache keys for each chunk
    keys = [
        make_key(
            ch,
            speaker,
            language,
            engine_type=engine_type,
            model_id=model_id,
            settings_hash=settings_hash,
        )
        for ch in cr.chunks
    ]

    return {
        "text_len": len(text),
        "chunks": len(cr.chunks),
        "speaker": speaker,
        "language": language,
        "device": device,
        "keys": keys,
    }


def _resolve_engine_meta(settings) -> tuple[str, str, str]:
    """
    Extract engine metadata from settings for cache key generation.

    Resolves the engine type (with alias handling), model ID, and generates
    a hash of engine-specific settings for cache invalidation.

    Args:
        settings: Loaded application settings.

    Returns:
        Tuple of (engine_type, model_id, settings_hash).
    """
    # Determine engine type from environment or settings
    engine_type = (os.getenv("TTS_MODEL_TYPE") or
                   settings.engine_type or "legacy").strip().lower()

    # Handle engine aliases (xtts -> legacy)
    aliases = {"xtts": "legacy", "xtts_v2": "legacy", "legacy": "legacy"}
    engine_type = aliases.get(engine_type, engine_type)

    tts = settings.raw.get("tts", {})

    # Extract model ID and settings based on engine type
    if engine_type == "legacy":
        model_id = settings.model_name
        engine_settings = {
            "engine": engine_type,
            "model_name": settings.model_name,
            "split_sentences": tts.get("split_sentences"),
            "quality": tts.get("quality", {}),
        }
    else:
        # Try multiple keys for model identification
        model_id = str(
            tts.get(engine_type, {}).get("model_id") or
            tts.get(engine_type, {}).get("checkpoint_path") or
            tts.get(engine_type, {}).get("model_path") or
            engine_type
        )
        engine_settings = {
            "engine": engine_type,
            "quality": tts.get("quality", {}),
            "engine_settings": tts.get(engine_type, {}),
        }

    return engine_type, model_id, hash_dict(engine_settings)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Orchestrates the CLI workflow:
        1. Parse arguments and configure environment
        2. Handle engine management commands (--engines, --setup)
        3. Load settings and resolve parameters
        4. Handle dry-run mode (if requested)
        5. Initialize engine and synthesize texts
        6. Output results in text or JSON format

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = _parse_args(argv)

    # Handle engine management commands first (no settings needed)
    if args.engines:
        from tts_ms.core.engine_setup import print_engine_status
        print_engine_status()
        return 0

    if args.setup:
        from tts_ms.core.engine_setup import get_engine_info, setup_engine
        engine_info = get_engine_info(args.setup)
        if not engine_info:
            print(f"Unknown engine: {args.setup}")
            print("Available engines: piper, f5tts, styletts2, legacy, chatterbox, cosyvoice, kokoro, qwen3tts, vibevoice")
            return 1

        print(f"\nSetting up {engine_info.display_name}...")
        result = setup_engine(args.setup, auto_install=args.auto_install)

        if result.satisfied:
            print(f"[OK] {result.message}")
            return 0
        else:
            print(f"[FAILED] {result.message}")
            return 1

    # Apply device override to environment
    if args.device:
        os.environ["TTS_DEVICE"] = args.device

    # Initialize logging and set request ID for tracing
    configure_logging()
    log = get_logger("tts-ms.cli")
    set_request_id(str(uuid4())[:12])

    # Load settings and resolve TTS parameters
    settings = load_settings("config/settings.yaml")
    speaker = args.speaker or os.getenv("TTS_MS_SPEAKER") or settings.default_speaker
    language = args.language or os.getenv("TTS_MS_LANGUAGE") or settings.default_language
    device = args.device or os.getenv("TTS_DEVICE") or settings.device

    # Prepare input texts and output paths
    texts = _load_texts(args)
    out_paths = _resolve_output_paths(args, len(texts))

    # Get engine metadata for cache keying
    engine_type, model_id, settings_hash = _resolve_engine_meta(settings)

    # Handle dry-run mode: analyze without synthesis
    if args.dry_run:
        summaries = [
            _summary_for_text(t, speaker, language, device, engine_type, model_id, settings_hash)
            for t in texts
        ]
        payload = {"ok": True, "dry_run": True, "items": summaries}

        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            info(log, "dry_run", items=len(texts), speaker=speaker,
                 language=language, device=device)
            print(payload)
        print("DRY_RUN_OK")
        return 0

    # Actual synthesis mode
    from tts_ms.tts.engine import get_engine
    engine = get_engine(settings)
    results = []

    # Process each text and save output
    for text, out_path in zip(texts, out_paths):
        info(log, "synth_start", chars=len(text), out=str(out_path))

        res = engine.synthesize(
            text=text,
            speaker=speaker,
            language=language,
            split_sentences=None,
        )

        out_path.write_bytes(res.wav_bytes)
        results.append({
            "out": str(out_path),
            "bytes": len(res.wav_bytes),
            "sample_rate": res.sample_rate
        })

    # Output final results
    payload = {"ok": True, "dry_run": False, "items": results}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(payload)
    print("CLI_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
