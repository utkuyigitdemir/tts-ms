#!/usr/bin/env python3
"""
Full Multi-Engine TTS Benchmark Script.

Tests all 9 TTS engines via Docker containers with interview questions
in Turkish and English. Each question/answer pair is synthesized separately.

Usage:
    python scripts/full_benchmark.py                          # All engines
    python scripts/full_benchmark.py --engines piper kokoro   # Specific engines
    python scripts/full_benchmark.py --dry-run                # Show plan only
    python scripts/full_benchmark.py --skip-build             # Use existing images

Requires: httpx (pip install httpx)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install with: pip install httpx")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Interview questions — split from voice_scripts_*.txt format
# ---------------------------------------------------------------------------

QUESTIONS_TR = [
    (
        "Sizi neden işe almalıyız?",
        "Ben bu pozisyon için en uygun adayım çünkü hem teknik bilgiye hem de takım çalışmasına yatkınlığa sahibim. Önceki projelerimde başarılı sonuçlar elde ettim ve şirketinize değer katabileceğime inanıyorum.",
    ),
    (
        "Siz bizi neden seçtiniz?",
        "Şirketinizin inovatif yaklaşımı ve sektördeki öncül konumu beni çok etkiledi. Ayrıca çalışanlarına verdiği değer ve kariyer gelişim fırsatları da tercihimde önemli rol oynadı.",
    ),
    (
        "Kendinizde kötü gördüğünüz özellikler nelerdir?",
        "Bazen aşırı detaycı olabiliyorum ve bu projelerin tamamlanma süresini uzatabiliyor. Ancak bu özelliğimi daha iyi yönetmek için zaman yönetimi teknikleri uyguluyorum.",
    ),
    (
        "Beş yıl sonra kendinizi nerede görüyorsunuz?",
        "Beş yıl sonra bu alanda uzmanlaşmış ve takım liderliğine geçmiş biri olarak kendimi görüyorum. Ayrıca sektörün gelişimine katkıda bulunmak istiyorum.",
    ),
    (
        "Maaş beklentiniz nedir?",
        "Piyasa koşullarını ve pozisyonun gerekliliklerini dikkate alarak, tecrübeme uygun adil bir ücret bekliyorum. Rakam konusunda esneğim ve şirketin bütçesine göre değerlendirmeye açığım.",
    ),
]

QUESTIONS_EN = [
    (
        "Why should we hire you?",
        "I am the most suitable candidate for this position because I have both technical knowledge and a tendency for teamwork. I achieved successful results in my previous projects and I believe I can add value to your company.",
    ),
    (
        "Why did you choose us?",
        "Your company's innovative approach and leading position in the industry impressed me greatly. Additionally, the value you place on your employees and career development opportunities played an important role in my choice.",
    ),
    (
        "What are the qualities you see as weaknesses in yourself?",
        "Sometimes I can be overly detail-oriented and this can extend the completion time of projects. However, I am applying time management techniques to better manage this trait.",
    ),
    (
        "Where do you see yourself in five years?",
        "In five years, I see myself as someone who has specialized in this field and transitioned into team leadership. I also want to contribute to the development of the industry.",
    ),
    (
        "What is your salary expectation?",
        "Taking into account market conditions and the requirements of the position, I expect a fair salary appropriate to my experience. I am flexible on the figure and open to evaluation based on the company's budget.",
    ),
]

# ---------------------------------------------------------------------------
# Engine configuration (CPU-only, no GPU)
# ---------------------------------------------------------------------------

# CPU count for normalizing psutil's per-process cpu_percent (can exceed 100%
# on multi-core) to a 0-100% scale.  Docker without --cpus exposes all host
# CPUs, so host cpu_count matches what the container sees.
_CPU_COUNT = os.cpu_count() or 1


@dataclass
class EngineConfig:
    name: str
    health_timeout: int       # seconds
    synth_timeout: int        # seconds
    native_cpu: bool = False
    languages: tuple[str, ...] = ("tr", "en")  # supported languages


ENGINE_CONFIGS: list[EngineConfig] = [
    EngineConfig("piper",      health_timeout=60,  synth_timeout=60,  native_cpu=True,  languages=("tr", "en")),
    EngineConfig("kokoro",     health_timeout=60,  synth_timeout=60,  native_cpu=True,  languages=("en",)),  # no TR model, falls back to en-us
    EngineConfig("legacy",     health_timeout=300, synth_timeout=300, languages=("tr", "en")),
    EngineConfig("f5tts",      health_timeout=300, synth_timeout=300, languages=("tr", "en")),
    EngineConfig("styletts2",  health_timeout=300, synth_timeout=300, languages=("en",)),
    EngineConfig("chatterbox", health_timeout=300, synth_timeout=300, languages=("tr", "en")),
    EngineConfig("cosyvoice",  health_timeout=360, synth_timeout=300, languages=("tr", "en")),
    EngineConfig("qwen3tts",   health_timeout=360, synth_timeout=300, languages=("tr", "en")),
    EngineConfig("vibevoice",  health_timeout=360, synth_timeout=600, languages=("tr", "en")),
]

ENGINE_MAP = {ec.name: ec for ec in ENGINE_CONFIGS}
ALL_ENGINES = [ec.name for ec in ENGINE_CONFIGS]

# ---------------------------------------------------------------------------
# Test item definition
# ---------------------------------------------------------------------------

@dataclass
class TestItem:
    """A single synthesis test."""
    lang: str          # "tr" or "en"
    index: int         # 1-based test number
    part: str          # "soru"/"cevap" or "question"/"answer"
    text: str          # text to synthesize
    filename: str      # base filename without extension


@dataclass
class TestResult:
    item: TestItem
    status: str = "PENDING"        # OK, FAIL, TIMEOUT, CONTAINER_DIED, SKIPPED
    duration: float = 0.0          # synthesis wall-clock seconds
    wav_size: int = 0              # bytes
    audio_duration_s: float = 0.0  # audio playback duration in seconds
    cpu_percent: float | None = None
    ram_delta_mb: float | None = None
    gpu_percent: float | None = None
    gpu_vram_delta_mb: float | None = None
    error: str = ""


@dataclass
class EngineSummary:
    engine: str
    status: str = "OK"             # OK, BUILD_FAILED, HEALTH_TIMEOUT
    startup_time: float = 0.0
    results: list[TestResult] = field(default_factory=list)
    error: str = ""


def build_test_items() -> list[TestItem]:
    """Build the full list of 20 test items (10 TR + 10 EN)."""
    items: list[TestItem] = []

    for i, (q, a) in enumerate(QUESTIONS_TR, 1):
        items.append(TestItem("tr", i, "soru",  q, f"test{i}_soru"))
        items.append(TestItem("tr", i, "cevap", a, f"test{i}_cevap"))

    for i, (q, a) in enumerate(QUESTIONS_EN, 1):
        items.append(TestItem("en", i, "question", q, f"test{i}_question"))
        items.append(TestItem("en", i, "answer",   a, f"test{i}_answer"))

    return items


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

CONTAINER_NAME = "tts-benchmark-{engine}"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run subprocess with UTF-8 encoding (avoids Windows cp1254 errors)."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    return subprocess.run(cmd, **kwargs)


def docker_build(engine: str) -> tuple[bool, str]:
    """Build Docker image for an engine. Returns (success, output)."""
    tag = f"tts-ms:{engine}"
    cmd = ["docker", "build", "--build-arg", f"TTS_MODEL_TYPE={engine}", "-t", tag, "."]
    print(f"  Building {tag} ...")
    try:
        result = _run(cmd, timeout=1800)
        if result.returncode != 0:
            return False, result.stderr[-2000:] if result.stderr else "Unknown build error"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Docker build timed out (30 min)"
    except Exception as e:
        return False, str(e)


def docker_run(engine: str, port: int, model_cache: str | None = None) -> tuple[bool, str]:
    """Start container. Returns (success, error)."""
    name = CONTAINER_NAME.format(engine=engine)
    tag = f"tts-ms:{engine}"

    # Clean up any existing container with the same name
    _run(["docker", "rm", "-f", name], timeout=30)

    cmd = [
        "docker", "run", "-d",
        "--name", name,
        "-p", f"{port}:8000",
        "-e", "TTS_MS_LOG_LEVEL=4",
        "-e", "TTS_MS_RESOURCES_ENABLED=1",
        "-e", "TTS_MS_RESOURCES_PER_STAGE=1",
        "-e", "TTS_MS_RESOURCES_SUMMARY=1",
        "-e", "TTS_MS_NO_COLOR=1",
    ]
    # Mount model cache volume to persist downloads between runs
    if model_cache:
        cache_path = str(Path(model_cache).resolve())
        cmd.extend(["-v", f"{cache_path}:/root/.cache", "-e", "TTS_HOME=/root/.cache/tts"])
    cmd.append(tag)
    try:
        result = _run(cmd, timeout=30)
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, ""
    except Exception as e:
        return False, str(e)


def docker_stop(engine: str) -> None:
    """Stop and remove container."""
    name = CONTAINER_NAME.format(engine=engine)
    _run(["docker", "stop", name], timeout=30)
    _run(["docker", "rm", "-f", name], timeout=30)


def docker_logs_since(engine: str, since: str) -> str:
    """Get container logs since a timestamp."""
    name = CONTAINER_NAME.format(engine=engine)
    try:
        result = _run(["docker", "logs", "--since", since, name], timeout=30)
        return result.stdout + result.stderr
    except Exception:
        return ""


def docker_logs_all(engine: str) -> str:
    """Get all container logs."""
    name = CONTAINER_NAME.format(engine=engine)
    try:
        result = _run(["docker", "logs", name], timeout=30)
        return result.stdout + result.stderr
    except Exception:
        return ""


def is_container_running(engine: str) -> bool:
    """Check if the container is still running."""
    name = CONTAINER_NAME.format(engine=engine)
    try:
        result = _run(["docker", "inspect", "-f", "{{.State.Running}}", name], timeout=10)
        return result.stdout.strip() == "true"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Health polling
# ---------------------------------------------------------------------------

def wait_for_health(port: int, timeout: int) -> tuple[bool, float]:
    """Poll /health until ready or timeout. Returns (ready, elapsed)."""
    url = f"http://localhost:{port}/health"
    start = time.monotonic()
    deadline = start + timeout

    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    # API returns {"ok": true, "warmed_up": true, ...}
                    if data.get("ok") and data.get("warmed_up", False):
                        return True, time.monotonic() - start
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            pass
        except Exception:
            pass
        time.sleep(3)

    return False, time.monotonic() - start


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(port: int, text: str, language: str, timeout: int) -> tuple[bytes | None, float, str]:
    """
    Call /v1/tts and return (wav_bytes, duration, error).
    """
    url = f"http://localhost:{port}/v1/tts"
    payload = {"text": text, "language": language}
    start = time.monotonic()

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
        elapsed = time.monotonic() - start

        if resp.status_code == 200:
            content = resp.content
            # Basic WAV validation: check RIFF header
            if len(content) >= 4 and content[:4] == b"RIFF":
                return content, elapsed, ""
            return content, elapsed, "Response is not valid WAV"
        else:
            try:
                err = resp.json()
                msg = err.get("message", resp.text[:200])
            except Exception:
                msg = resp.text[:200]
            return None, elapsed, f"HTTP {resp.status_code}: {msg}"

    except httpx.ReadTimeout:
        return None, time.monotonic() - start, f"Synthesis timeout ({timeout}s)"
    except httpx.ConnectError:
        return None, time.monotonic() - start, "Connection refused (container may have crashed)"
    except Exception as e:
        return None, time.monotonic() - start, str(e)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_RESOURCE_RE = re.compile(
    r"resources_summary\s+"
    r"(?P<pairs>(?:\w+=\S+\s*)+)"
)
_KV_RE = re.compile(r"(\w+)=([\d.+-]+)")


def parse_resources(log_text: str) -> dict[str, float]:
    """Parse resource metrics from container logs.

    cpu_percent is normalized from per-process (0..N*100) to system-wide
    (0..100) by dividing by the host CPU count.
    """
    resources: dict[str, float] = {}
    for match in _RESOURCE_RE.finditer(log_text):
        pairs = match.group("pairs")
        for kv in _KV_RE.finditer(pairs):
            key, val = kv.group(1), kv.group(2)
            try:
                resources[key] = float(val)
            except ValueError:
                pass
    # Normalize cpu_percent to 0-100% scale
    if "cpu_percent" in resources:
        resources["cpu_percent"] = resources["cpu_percent"] / _CPU_COUNT
    return resources


# ---------------------------------------------------------------------------
# WAV info helper
# ---------------------------------------------------------------------------

def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Get approximate duration of a WAV file in seconds."""
    if len(wav_bytes) < 44:
        return 0.0
    try:
        # Parse WAV header for sample rate and data size
        sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
        byte_rate = struct.unpack_from("<I", wav_bytes, 28)[0]
        if byte_rate == 0:
            return 0.0
        data_size = len(wav_bytes) - 44
        return data_size / byte_rate
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

@dataclass
class AggregateStats:
    """Aggregated resource statistics from a list of TestResults."""
    avg_cpu: float = 0.0
    peak_cpu: float = 0.0
    avg_ram: float = 0.0
    peak_ram: float = 0.0
    avg_gpu: float | None = None
    peak_gpu: float | None = None
    avg_gpu_vram: float | None = None
    peak_gpu_vram: float | None = None
    has_gpu: bool = False


def compute_aggregate_stats(results: list[TestResult]) -> AggregateStats:
    """Compute avg/peak statistics from a list of TestResults."""
    s = AggregateStats()

    cpu_vals = [r.cpu_percent for r in results if r.cpu_percent is not None]
    ram_vals = [r.ram_delta_mb for r in results if r.ram_delta_mb is not None]
    gpu_vals = [r.gpu_percent for r in results if r.gpu_percent is not None]
    vram_vals = [r.gpu_vram_delta_mb for r in results if r.gpu_vram_delta_mb is not None]

    if cpu_vals:
        s.avg_cpu = sum(cpu_vals) / len(cpu_vals)
        s.peak_cpu = max(cpu_vals)
    if ram_vals:
        s.avg_ram = sum(ram_vals) / len(ram_vals)
        s.peak_ram = max(ram_vals)
    if gpu_vals:
        s.has_gpu = True
        s.avg_gpu = sum(gpu_vals) / len(gpu_vals)
        s.peak_gpu = max(gpu_vals)
    if vram_vals:
        s.avg_gpu_vram = sum(vram_vals) / len(vram_vals)
        s.peak_gpu_vram = max(vram_vals)

    return s


def _format_stats_lines(s: AggregateStats) -> list[str]:
    """Format aggregate stats as summary lines."""
    lines = [
        f"  Avg CPU:      {s.avg_cpu:.1f}%",
        f"  Peak CPU:     {s.peak_cpu:.1f}%",
        f"  Avg RAM:      {s.avg_ram:+.1f} MB",
        f"  Peak RAM:     {s.peak_ram:+.1f} MB",
    ]
    if s.has_gpu:
        lines.append(f"  Avg GPU:      {s.avg_gpu:.1f}%")
        lines.append(f"  Peak GPU:     {s.peak_gpu:.1f}%")
        if s.avg_gpu_vram is not None:
            lines.append(f"  Avg VRAM:     {s.avg_gpu_vram:+.1f} MB")
        if s.peak_gpu_vram is not None:
            lines.append(f"  Peak VRAM:    {s.peak_gpu_vram:+.1f} MB")
    else:
        lines.append("  GPU:          not used (no CUDA device detected)")
    return lines


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

def write_engine_summary(summary: EngineSummary, results_dir: Path) -> None:
    """Write per-engine summary.txt."""
    engine_dir = results_dir / summary.engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    path = engine_dir / "summary.txt"
    cfg = ENGINE_MAP.get(summary.engine)
    native_langs = cfg.languages if cfg else ("tr", "en")

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"ENGINE: {summary.engine}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if summary.status != "OK":
        lines.append(f"Status: {summary.status}")
        if summary.error:
            lines.append(f"Error: {summary.error}")
        lines.append("=" * 60)
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append(f"Startup time: {summary.startup_time:.1f}s")
    lines.append(f"Native languages: {', '.join(native_langs)}")
    lines.append("=" * 60)

    # Group results by language
    for lang, label in [("tr", "TURKISH (tr)"), ("en", "ENGLISH (en)")]:
        lang_results = [r for r in summary.results if r.item.lang == lang]
        if not lang_results:
            if lang not in native_langs:
                lines.append("")
                lines.append(f"{label}  — no support, skipped")
            continue
        lines.append("")
        lines.append(label)
        lines.append("-" * 60)
        for r in lang_results:
            size_kb = r.wav_size / 1024
            cpu_str = f"CPU: {r.cpu_percent:5.1f}%" if r.cpu_percent is not None else "CPU:     -"
            ram_str = f"RAM: {r.ram_delta_mb:+.1f} MB" if r.ram_delta_mb is not None else "RAM: -"
            gpu_str = ""
            if r.gpu_percent is not None:
                gpu_str = f" | GPU: {r.gpu_percent:5.1f}%"
                if r.gpu_vram_delta_mb is not None:
                    gpu_str += f" | VRAM: {r.gpu_vram_delta_mb:+.1f} MB"
            dur_str = f"[{r.audio_duration_s:5.1f}s]" if r.audio_duration_s > 0 else "[    -]"
            lines.append(
                f"  {r.item.filename:<18} {r.duration:5.2f}s | "
                f"{size_kb:7.1f} KB | {dur_str} | {cpu_str} | {ram_str}{gpu_str} | {r.status}"
            )
            if r.error:
                lines.append(f"    ERROR: {r.error[:100]}")

    # Totals
    all_results = summary.results
    passed = sum(1 for r in all_results if r.status == "OK")
    failed = len(all_results) - passed
    total_time = sum(r.duration for r in all_results)
    total_audio = sum(r.wav_size for r in all_results)
    avg_time = total_time / len(all_results) if all_results else 0
    total_audio_dur = sum(r.audio_duration_s for r in all_results)
    stats = compute_aggregate_stats(all_results)

    lines.append("")
    lines.append("TOTALS")
    lines.append("-" * 60)
    lines.append(f"  Tests:        {len(all_results)} total | {passed} pass | {failed} fail")
    lines.append(f"  Total time:   {total_time:.2f}s (synthesis wall-clock)")
    lines.append(f"  Avg time:     {avg_time:.2f}s")
    lines.append(f"  Total audio:  {total_audio / (1024*1024):.1f} MB ({total_audio_dur:.1f}s playback)")
    lines.extend(_format_stats_lines(stats))

    path.write_text("\n".join(lines), encoding="utf-8")

    # Save machine-readable JSON for global summary aggregation
    json_path = engine_dir / "results.json"
    json_data = {
        "engine": summary.engine,
        "status": summary.status,
        "startup_time": summary.startup_time,
        "error": summary.error,
        "results": [
            {
                "lang": r.item.lang,
                "filename": r.item.filename,
                "status": r.status,
                "duration": r.duration,
                "wav_size": r.wav_size,
                "audio_duration_s": r.audio_duration_s,
                "cpu_percent": r.cpu_percent,
                "ram_delta_mb": r.ram_delta_mb,
                "gpu_percent": r.gpu_percent,
                "gpu_vram_delta_mb": r.gpu_vram_delta_mb,
                "error": r.error,
            }
            for r in summary.results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_all_engine_summaries(results_dir: Path) -> list[EngineSummary]:
    """Scan results_dir for all engine results.json and reconstruct summaries."""
    summaries: list[EngineSummary] = []
    # Iterate in canonical engine order, then any extras
    known = [ec.name for ec in ENGINE_CONFIGS]
    dirs = sorted(
        (d for d in results_dir.iterdir() if d.is_dir() and (d / "results.json").exists()),
        key=lambda d: (known.index(d.name) if d.name in known else 999, d.name),
    )
    for engine_dir in dirs:
        json_path = engine_dir / "results.json"
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        es = EngineSummary(
            engine=data["engine"],
            status=data.get("status", "OK"),
            startup_time=data.get("startup_time", 0),
            error=data.get("error", ""),
        )
        for rd in data.get("results", []):
            es.results.append(TestResult(
                item=TestItem(
                    lang=rd["lang"],
                    index=0,
                    part="",
                    text="",
                    filename=rd["filename"],
                ),
                status=rd.get("status", "OK"),
                duration=rd.get("duration", 0),
                wav_size=rd.get("wav_size", 0),
                audio_duration_s=rd.get("audio_duration_s", 0),
                cpu_percent=rd.get("cpu_percent"),
                ram_delta_mb=rd.get("ram_delta_mb"),
                gpu_percent=rd.get("gpu_percent"),
                gpu_vram_delta_mb=rd.get("gpu_vram_delta_mb"),
                error=rd.get("error", ""),
            ))
        summaries.append(es)
    return summaries


def write_global_summary(summaries: list[EngineSummary], results_dir: Path) -> None:
    """Write global results/summary.txt by scanning ALL engine results in results_dir."""
    path = results_dir / "summary.txt"

    # Merge current run summaries with previously saved ones
    all_summaries = _load_all_engine_summaries(results_dir)
    # Override with current run data (fresher)
    existing_names = {s.engine for s in all_summaries}
    for s in summaries:
        if s.engine not in existing_names:
            all_summaries.append(s)
    summaries = all_summaries

    # Check if any engine has GPU data
    any_gpu = any(
        r.gpu_percent is not None
        for s in summaries
        for r in s.results
    )

    lines: list[str] = []
    lines.append("MULTI-ENGINE BENCHMARK SUMMARY")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"CPU cores: {_CPU_COUNT}")
    lines.append("=" * 120)

    # Build table header dynamically based on GPU availability
    hdr = (
        f"| {'Engine':<12} | {'Status':^14} | {'Tests':^5} | {'Pass':^4} | "
        f"{'Avg Time':^8} | {'Total':^8} | {'Audio':^8} | "
        f"{'Avg CPU':^7} | {'Peak CPU':^8} | {'Avg RAM':^10} | {'Peak RAM':^10} |"
    )
    sep = (
        f"|{'-'*14}|{'-'*16}|{'-'*7}|{'-'*6}|"
        f"{'-'*10}|{'-'*10}|{'-'*10}|"
        f"{'-'*9}|{'-'*10}|{'-'*12}|{'-'*12}|"
    )
    if any_gpu:
        hdr += f" {'Avg GPU':^7} | {'Peak GPU':^8} |"
        sep += f"{'-'*9}|{'-'*10}|"

    lines.append(hdr)
    lines.append(sep)

    for s in summaries:
        if s.status != "OK":
            dash = "   -"
            row = (
                f"| {s.engine:<12} | {s.status:^14} | {dash:^5} | {dash:^4} | "
                f"{dash:^8} | {dash:^8} | {dash:^8} | "
                f"{dash:^7} | {dash:^8} | {dash:^10} | {dash:^10} |"
            )
            if any_gpu:
                row += f" {dash:^7} | {dash:^8} |"
            lines.append(row)
            continue

        total = len(s.results)
        passed = sum(1 for r in s.results if r.status == "OK")
        total_time = sum(r.duration for r in s.results)
        avg_time = total_time / total if total else 0
        total_audio_dur = sum(r.audio_duration_s for r in s.results)
        st = compute_aggregate_stats(s.results)

        row = (
            f"| {s.engine:<12} | {'OK':^14} | {total:^5} | {passed:^4} | "
            f"{avg_time:>6.2f}s  | {total_time:>6.1f}s  | {total_audio_dur:>6.1f}s  | "
            f"{st.avg_cpu:>5.1f}%  | {st.peak_cpu:>6.1f}%  | {st.avg_ram:>+8.1f} MB | {st.peak_ram:>+8.1f} MB |"
        )
        if any_gpu:
            avg_g = f"{st.avg_gpu:.1f}%" if st.avg_gpu is not None else "-"
            peak_g = f"{st.peak_gpu:.1f}%" if st.peak_gpu is not None else "-"
            row += f" {avg_g:>5}   | {peak_g:>6}   |"
        lines.append(row)

    lines.append("")
    if not any_gpu:
        lines.append("NOTE: GPU not used (no CUDA device detected). All engines ran on CPU.")
        lines.append("")
    lines.append(f"Total engines tested: {len(summaries)}")
    lines.append(f"Successful: {sum(1 for s in summaries if s.status == 'OK')}")
    lines.append(f"Failed: {sum(1 for s in summaries if s.status != 'OK')}")

    # Per-engine detail blocks
    for s in summaries:
        if s.status != "OK":
            continue
        st = compute_aggregate_stats(s.results)
        total_time = sum(r.duration for r in s.results)
        total_audio_dur = sum(r.audio_duration_s for r in s.results)
        total_audio = sum(r.wav_size for r in s.results)
        passed = sum(1 for r in s.results if r.status == "OK")

        cfg = ENGINE_MAP.get(s.engine)
        native_langs = cfg.languages if cfg else ("tr", "en")
        skipped = [l for l in ("tr", "en") if l not in native_langs]

        lines.append("")
        lines.append(f"--- {s.engine} ---")
        lines.append(f"  Startup:      {s.startup_time:.1f}s")
        lines.append(f"  Languages:    {', '.join(native_langs)}")
        if skipped:
            lines.append(f"  Skipped:      {', '.join(skipped)} (no support)")
        lines.append(f"  Tests:        {len(s.results)} total | {passed} pass")
        lines.append(f"  Synth time:   {total_time:.2f}s total | {total_time / len(s.results):.2f}s avg")
        lines.append(f"  Audio:        {total_audio / (1024*1024):.1f} MB | {total_audio_dur:.1f}s playback")
        lines.extend(_format_stats_lines(st))

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(engines: list[str]) -> None:
    """Print the benchmark plan without executing."""
    items = build_test_items()
    print("\n=== DRY RUN — Benchmark Plan ===\n")
    print(f"Engines: {', '.join(engines)}")
    print(f"Tests per engine: {len(items)}")
    print(f"Total tests: {len(items) * len(engines)}")
    print(f"CPU cores: {_CPU_COUNT} (used for CPU% normalization)")
    print()

    for engine in engines:
        cfg = ENGINE_MAP[engine]
        native = ", ".join(cfg.languages)
        print(
            f"  [{engine:<12}] native_langs={native:<6}  "
            f"health={cfg.health_timeout}s  synth={cfg.synth_timeout}s"
        )

    print("\nTest items:")
    for item in items:
        label = f"[{item.lang}] test{item.index}_{item.part}"
        preview = item.text[:60] + ("..." if len(item.text) > 60 else "")
        print(f"  {label:<28} {preview}")

    print("\nNo API calls will be made.")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    engines: list[str],
    skip_build: bool,
    results_dir: Path,
    port: int,
    timeout_multiplier: float,
    model_cache: str | None = None,
) -> list[EngineSummary]:
    """Run the full benchmark for selected engines."""
    summaries: list[EngineSummary] = []

    print(f"\n{'='*60}")
    print(f"  FULL MULTI-ENGINE TTS BENCHMARK")
    print(f"  Engines: {', '.join(engines)}")
    print(f"  Tests per engine: 20 (10 TR + 10 EN)")
    print(f"  CPU cores: {_CPU_COUNT} (for CPU% normalization)")
    print(f"  Results directory: {results_dir}")
    print(f"{'='*60}\n")

    all_items = build_test_items()

    for eng_idx, engine in enumerate(engines, 1):
        cfg = ENGINE_MAP[engine]
        health_timeout = int(cfg.health_timeout * timeout_multiplier)
        synth_timeout = int(cfg.synth_timeout * timeout_multiplier)
        summary = EngineSummary(engine=engine)

        # Filter test items: only run languages the engine supports
        items = [it for it in all_items if it.lang in cfg.languages]
        skipped_langs = [l for l in ("tr", "en") if l not in cfg.languages]

        native = ", ".join(cfg.languages)
        skip_note = f"  (skipping: {', '.join(skipped_langs)})" if skipped_langs else ""
        print(f"\n[{eng_idx}/{len(engines)}] ENGINE: {engine}  (native: {native}, tests: {len(items)}){skip_note}")
        print(f"  Health timeout: {health_timeout}s  Synth timeout: {synth_timeout}s")
        print("-" * 60)

        # --- Docker build ---
        if not skip_build:
            ok, err = docker_build(engine)
            if not ok:
                print(f"  BUILD FAILED: {err[:200]}")
                summary.status = "BUILD_FAILED"
                summary.error = err[:500]
                summaries.append(summary)
                write_engine_summary(summary, results_dir)
                continue
            print("  Build: OK")
        else:
            print("  Build: skipped (--skip-build)")

        # --- Docker run ---
        try:
            ok, err = docker_run(engine, port, model_cache=model_cache)
            if not ok:
                print(f"  RUN FAILED: {err}")
                summary.status = "RUN_FAILED"
                summary.error = err
                summaries.append(summary)
                write_engine_summary(summary, results_dir)
                continue
            print(f"  Container started on port {port}")

            # --- Health check ---
            print(f"  Waiting for health (up to {health_timeout}s) ...")
            ready, startup = wait_for_health(port, health_timeout)

            if not ready:
                print(f"  HEALTH TIMEOUT after {startup:.1f}s")
                # Save startup logs
                startup_logs = docker_logs_all(engine)
                log_dir = results_dir / engine
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "startup_failure.log").write_text(startup_logs, encoding="utf-8")
                summary.status = "HEALTH_TIMEOUT"
                summary.startup_time = startup
                summary.error = f"Health check failed after {startup:.0f}s"
                summaries.append(summary)
                write_engine_summary(summary, results_dir)
                continue

            summary.startup_time = startup
            print(f"  Health: OK ({startup:.1f}s)")

            # --- Run tests ---
            for test_idx, item in enumerate(items, 1):
                # Check container is still alive
                if not is_container_running(engine):
                    print(f"  CONTAINER DIED during test {test_idx}")
                    # Mark remaining tests
                    for remaining in items[test_idx - 1:]:
                        summary.results.append(TestResult(
                            item=remaining,
                            status="CONTAINER_DIED",
                            error="Container stopped unexpectedly",
                        ))
                    break

                # Ensure output directory exists
                out_dir = results_dir / engine / item.lang
                out_dir.mkdir(parents=True, exist_ok=True)

                # Record timestamp for log extraction
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

                # Synthesize
                label = f"[{item.lang}] {item.filename}"
                print(f"  ({test_idx:2}/{len(items)}) {label:<30}", end="", flush=True)

                wav_bytes, duration, err = synthesize(port, item.text, item.lang, synth_timeout)

                result = TestResult(item=item, duration=duration)

                if wav_bytes and not err:
                    result.status = "OK"
                    result.wav_size = len(wav_bytes)
                    result.audio_duration_s = wav_duration_seconds(wav_bytes)

                    # Save WAV
                    wav_path = out_dir / f"{item.filename}.wav"
                    wav_path.write_bytes(wav_bytes)

                    # Small delay to let logs flush
                    time.sleep(0.5)

                    # Collect and save logs for this test
                    logs = docker_logs_since(engine, timestamp)
                    log_path = out_dir / f"{item.filename}.log"
                    log_path.write_text(logs, encoding="utf-8")

                    # Parse resources from logs
                    resources = parse_resources(logs)
                    result.cpu_percent = resources.get("cpu_percent")
                    result.ram_delta_mb = resources.get("ram_delta_mb")
                    result.gpu_percent = resources.get("gpu_percent")
                    result.gpu_vram_delta_mb = resources.get("gpu_vram_delta_mb")

                    size_kb = result.wav_size / 1024
                    cpu_str = f"CPU:{result.cpu_percent:.0f}%" if result.cpu_percent is not None else ""
                    aud_str = f"[{result.audio_duration_s:.1f}s]"
                    print(f" {duration:5.2f}s  {size_kb:6.1f}KB  {aud_str:>7}  {cpu_str}  OK")

                elif err and err.startswith("Synthesis timeout"):
                    result.status = "TIMEOUT"
                    result.error = err

                    # Save logs even on failure
                    time.sleep(0.5)
                    logs = docker_logs_since(engine, timestamp)
                    log_path = out_dir / f"{item.filename}.log"
                    log_path.write_text(logs, encoding="utf-8")

                    print(f" {duration:5.2f}s  TIMEOUT")

                else:
                    result.status = "FAIL"
                    result.error = err

                    # Save what we have
                    if wav_bytes:
                        wav_path = out_dir / f"{item.filename}.wav"
                        wav_path.write_bytes(wav_bytes)

                    time.sleep(0.5)
                    logs = docker_logs_since(engine, timestamp)
                    log_path = out_dir / f"{item.filename}.log"
                    log_path.write_text(logs, encoding="utf-8")

                    print(f" {duration:5.2f}s  FAIL: {err[:60]}")

                summary.results.append(result)

            # Print engine totals
            passed = sum(1 for r in summary.results if r.status == "OK")
            total = len(summary.results)
            total_time = sum(r.duration for r in summary.results)
            print(f"\n  Engine {engine}: {passed}/{total} passed, {total_time:.1f}s total")

        finally:
            # Always clean up container
            print(f"  Stopping container ...")
            docker_stop(engine)

        summaries.append(summary)
        write_engine_summary(summary, results_dir)

    # --- Global summary ---
    write_global_summary(summaries, results_dir)
    print(f"\n{'='*60}")
    print("  BENCHMARK COMPLETE")
    print(f"  Results: {results_dir}")
    print(f"  Summary: {results_dir / 'summary.txt'}")
    print(f"{'='*60}\n")

    # Print quick overview
    for s in summaries:
        if s.status == "OK":
            passed = sum(1 for r in s.results if r.status == "OK")
            total_time = sum(r.duration for r in s.results)
            print(f"  {s.engine:<12}  {s.status:>8}  {passed:>2}/{len(s.results)} passed  {total_time:>7.1f}s")
        else:
            print(f"  {s.engine:<12}  {s.status:>16}")

    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full Multi-Engine TTS Benchmark via Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/full_benchmark.py --dry-run
  python scripts/full_benchmark.py --engines piper kokoro
  python scripts/full_benchmark.py --skip-build --timeout-multiplier 2.0
  python scripts/full_benchmark.py --engines piper --results-dir my_results/
        """,
    )
    parser.add_argument(
        "--engines", nargs="+", choices=ALL_ENGINES, default=ALL_ENGINES,
        help=f"Engines to test (default: all). Choices: {', '.join(ALL_ENGINES)}",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip Docker build (use existing images)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Container port mapping (default: 8000)",
    )
    parser.add_argument(
        "--timeout-multiplier", type=float, default=1.0,
        help="Multiply all timeouts by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without executing",
    )
    parser.add_argument(
        "--model-cache", type=str, default=None,
        help="Host directory to cache model downloads (mounted as /root/.cache in container)",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.engines)
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_benchmark(
        engines=args.engines,
        skip_build=args.skip_build,
        results_dir=results_dir,
        port=args.port,
        timeout_multiplier=args.timeout_multiplier,
        model_cache=args.model_cache,
    )


if __name__ == "__main__":
    main()
