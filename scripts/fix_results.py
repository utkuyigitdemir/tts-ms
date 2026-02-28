"""Fix results: denormalize CPU%, remove TR for no-TR engines, regenerate summaries."""
import json
import os
from datetime import datetime

RESULTS_DIR = "results"
NO_TR_ENGINES = ["kokoro", "styletts2", "cosyvoice", "f5tts", "qwen3tts", "vibevoice"]
CPU_COUNT = 22


def fmt_kb(b):
    return f"{b/1024:.1f} KB"


def fmt_mb(b):
    return f"{b/(1024*1024):.1f} MB"


def regenerate_engine_summary(engine):
    engine_dir = os.path.join(RESULTS_DIR, engine)
    json_path = os.path.join(engine_dir, "results.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    startup = data.get("startup_time", 0)

    langs_in_results = sorted(set(r["lang"] for r in results))
    if engine in NO_TR_ENGINES:
        native_langs = ["en"]
    else:
        native_langs = langs_in_results if langs_in_results else ["tr", "en"]

    lines = []
    lines.append("=" * 60)
    lines.append(f"ENGINE: {engine}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Startup time: {startup:.1f}s")
    lines.append(f"Native languages: {', '.join(native_langs)}")
    lines.append("=" * 60)
    lines.append("")

    for lang_code, lang_label in [("tr", "TURKISH (tr)"), ("en", "ENGLISH (en)")]:
        lang_results = [r for r in results if r["lang"] == lang_code]

        if not lang_results:
            if engine in NO_TR_ENGINES and lang_code == "tr":
                lines.append(f"{lang_label}  — Türkçe desteklenmiyor, atlandı")
                lines.append("")
            continue

        lines.append(lang_label)
        lines.append("-" * 60)

        for r in lang_results:
            name = r["filename"]
            dur = r["duration"]
            wav_kb = fmt_kb(r["wav_size"]) if r["wav_size"] else "   -"
            audio_s = f'[{r["audio_duration_s"]:.1f}s]' if r.get("audio_duration_s") else "[   -]"
            cpu = f'CPU: {r["cpu_percent"]:.1f}%' if r.get("cpu_percent") is not None else "CPU:     -"
            ram = f'RAM: +{r["ram_delta_mb"]:.1f} MB' if r.get("ram_delta_mb") is not None else "RAM:     -"
            status = r["status"]
            err = r.get("error", "")

            if status == "OK":
                lines.append(f"  {name:<20s} {dur:6.2f}s | {wav_kb:>10s} | {audio_s:>7s} | {cpu:>14s} | {ram:>14s} | OK")
            else:
                lines.append(f"  {name:<20s} {dur:6.2f}s | {status}: {err[:60]}")

        lines.append("")

    ok_results = [r for r in results if r["status"] == "OK"]
    total_time = sum(r["duration"] for r in results)
    avg_time = total_time / len(results) if results else 0
    total_wav = sum(r.get("wav_size", 0) or 0 for r in ok_results)
    total_audio = sum(r.get("audio_duration_s", 0) or 0 for r in ok_results)

    cpu_vals = [r["cpu_percent"] for r in ok_results if r.get("cpu_percent") is not None]
    ram_vals = [r["ram_delta_mb"] for r in ok_results if r.get("ram_delta_mb") is not None]

    avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0
    peak_cpu = max(cpu_vals) if cpu_vals else 0
    avg_ram = sum(ram_vals) / len(ram_vals) if ram_vals else 0
    peak_ram = max(ram_vals) if ram_vals else 0

    lines.append("TOTALS")
    lines.append("-" * 60)
    fail_count = len(results) - len(ok_results)
    lines.append(f"  Tests:        {len(results)} total | {len(ok_results)} pass | {fail_count} fail")
    lines.append(f"  Total time:   {total_time:.2f}s (synthesis wall-clock)")
    lines.append(f"  Avg time:     {avg_time:.2f}s")
    lines.append(f"  Total audio:  {fmt_mb(total_wav)} ({total_audio:.1f}s playback)")
    lines.append(f"  Avg CPU:      {avg_cpu:.1f}%")
    lines.append(f"  Peak CPU:     {peak_cpu:.1f}%")
    ram_s = "+" if avg_ram >= 0 else ""
    lines.append(f"  Avg RAM:      {ram_s}{avg_ram:.1f} MB")
    lines.append(f"  Peak RAM:     +{peak_ram:.1f} MB")
    lines.append(f"  GPU:          not used (no CUDA device detected)")

    summary_path = os.path.join(engine_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[{engine}] summary.txt regenerated (tests={len(results)}, avg_cpu={avg_cpu:.1f}%)")
    return {
        "engine": engine,
        "status": data["status"],
        "tests": len(results),
        "passed": len(ok_results),
        "avg_time": avg_time,
        "total_time": total_time,
        "total_audio": total_audio,
        "avg_cpu": avg_cpu,
        "peak_cpu": peak_cpu,
        "avg_ram": avg_ram,
        "peak_ram": peak_ram,
        "startup": startup,
        "native_langs": native_langs,
        "total_wav": total_wav,
    }


def main():
    engine_order = ["piper", "kokoro", "legacy", "f5tts", "styletts2", "chatterbox", "cosyvoice", "qwen3tts", "vibevoice"]
    summaries = []
    for engine in engine_order:
        engine_dir = os.path.join(RESULTS_DIR, engine)
        if os.path.isdir(engine_dir) and os.path.exists(os.path.join(engine_dir, "results.json")):
            summaries.append(regenerate_engine_summary(engine))

    # Global summary
    lines = []
    lines.append("MULTI-ENGINE BENCHMARK SUMMARY")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("CPU cores: 22")
    lines.append("=" * 120)

    hdr = f'| {"Engine":<12s} | {"Status":^14s} | {"Tests":>5s} | {"Pass":>4s} | {"Avg Time":>8s} | {"Total":>8s} | {"Audio":>8s} | {"Avg CPU":>7s} | {"Peak CPU":>8s} | {"Avg RAM":>10s} | {"Peak RAM":>10s} |'
    sep = f'|{"-"*14}|{"-"*16}|{"-"*7}|{"-"*6}|{"-"*10}|{"-"*10}|{"-"*10}|{"-"*9}|{"-"*10}|{"-"*12}|{"-"*12}|'
    lines.append(hdr)
    lines.append(sep)

    for s in summaries:
        lines.append(
            f'| {s["engine"]:<12s} | {s["status"]:^14s} | {s["tests"]:>5d} | {s["passed"]:>4d} '
            f'| {s["avg_time"]:>7.2f}s | {s["total_time"]:>7.1f}s | {s["total_audio"]:>7.1f}s '
            f'| {s["avg_cpu"]:>6.1f}% | {s["peak_cpu"]:>7.1f}% '
            f'| {s["avg_ram"]:>+9.1f} MB | {s["peak_ram"]:>+9.1f} MB |'
        )

    lines.append("")
    lines.append("NOTE: GPU not used (no CUDA device detected). All engines ran on CPU.")
    lines.append(f"NOTE: CPU percentages are raw psutil values (0-{22*100}% range for {22} cores).")
    lines.append("")
    ok_count = sum(1 for s in summaries if s["status"] == "OK")
    fail_count = len(summaries) - ok_count
    lines.append(f"Total engines tested: {len(summaries)}")
    lines.append(f"Successful: {ok_count}")
    lines.append(f"Failed: {fail_count}")

    for s in summaries:
        lines.append("")
        lines.append(f'--- {s["engine"]} ---')
        lines.append(f'  Startup:      {s["startup"]:.1f}s')
        lines.append(f'  Languages:    {", ".join(s["native_langs"])}')
        if s["engine"] in NO_TR_ENGINES:
            lines.append("  Skipped:      tr (Türkçe desteklenmiyor)")
        lines.append(f'  Tests:        {s["tests"]} total | {s["passed"]} pass')
        lines.append(f'  Synth time:   {s["total_time"]:.2f}s total | {s["avg_time"]:.2f}s avg')
        lines.append(f'  Audio:        {s["total_wav"]/(1024*1024):.1f} MB | {s["total_audio"]:.1f}s playback')
        lines.append(f'  Avg CPU:      {s["avg_cpu"]:.1f}%')
        lines.append(f'  Peak CPU:     {s["peak_cpu"]:.1f}%')
        ram_sign = "+" if s["avg_ram"] >= 0 else ""
        lines.append(f'  Avg RAM:      {ram_sign}{s["avg_ram"]:.1f} MB')
        lines.append(f'  Peak RAM:     +{s["peak_ram"]:.1f} MB')
        lines.append("  GPU:          not used (no CUDA device detected)")

    global_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(global_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nGlobal summary.txt regenerated")
    print("All done.")


if __name__ == "__main__":
    main()
