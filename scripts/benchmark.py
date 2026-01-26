from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx


@dataclass
class BenchResult:
    cold_v1_s: float
    hot_v1_s: float
    stream_first_chunk_s: float
    stream_total_s: float
    stream_chunks: int
    bytes_v1: int


def _post_v1(client: httpx.Client, base_url: str, text: str) -> Tuple[float, int]:
    t0 = time.perf_counter()
    r = client.post(f"{base_url}/v1/tts", json={"text": text})
    dt = time.perf_counter() - t0
    r.raise_for_status()
    b = r.content
    # sanity header
    if not (b[:4] == b"RIFF" and b[8:12] == b"WAVE"):
        raise RuntimeError("v1 response is not WAV (RIFF/WAVE header missing)")
    return dt, len(b)


def _stream_sse_first_chunk_and_done(
    client: httpx.Client, base_url: str, text: str
) -> Tuple[float, float, int]:
    t0 = time.perf_counter()
    first_chunk_dt: Optional[float] = None
    chunks = 0
    done = False

    with client.stream("POST", f"{base_url}/v1/tts/stream", json={"text": text}) as r:
        r.raise_for_status()

        cur_event = None
        cur_data = None

        for raw_line in r.iter_lines():
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                # end of event
                if cur_event and cur_data:
                    if cur_event == "chunk":
                        if first_chunk_dt is None:
                            first_chunk_dt = time.perf_counter() - t0
                        payload = json.loads(cur_data)
                        _ = base64.b64decode(payload["audio_wav_b64"])  # sanity decode
                        chunks += 1
                    elif cur_event == "done":
                        done = True
                        break
                cur_event, cur_data = None, None
                continue

            if line.startswith("event:"):
                cur_event = line.split("event:", 1)[1].strip()
            elif line.startswith("data:"):
                cur_data = line.split("data:", 1)[1].strip()

    total = time.perf_counter() - t0
    if first_chunk_dt is None:
        raise RuntimeError("stream did not produce any chunk event")
    if not done:
        raise RuntimeError("stream did not produce done event")
    return first_chunk_dt, total, chunks


def run_benchmark(base_url: str, text: str, timeout_s: float = 120.0) -> BenchResult:
    with httpx.Client(timeout=timeout_s) as client:
        cold_s, nbytes = _post_v1(client, base_url, text)
        hot_s, _ = _post_v1(client, base_url, text)
        first_chunk_s, stream_total_s, chunks = _stream_sse_first_chunk_and_done(client, base_url, text)

    return BenchResult(
        cold_v1_s=cold_s,
        hot_v1_s=hot_s,
        stream_first_chunk_s=first_chunk_s,
        stream_total_s=stream_total_s,
        stream_chunks=chunks,
        bytes_v1=nbytes,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--text", default="Merhaba! Bu bir benchmark testidir. Aynı metni iki kez üreteceğim.")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print("BENCHMARK_DRY_RUN_OK")
        print({"base_url": args.base_url, "text_len": len(args.text), "timeout": args.timeout})
        return

    res = run_benchmark(args.base_url, args.text, timeout_s=args.timeout)

    print("BENCHMARK_OK")
    print(f"base_url: {args.base_url}")
    print(f"text_len: {len(args.text)}")
    print(f"v1 cold: {res.cold_v1_s:.3f}s | bytes={res.bytes_v1}")
    print(f"v1 hot : {res.hot_v1_s:.3f}s")
    print(f"stream first_chunk: {res.stream_first_chunk_s:.3f}s | chunks={res.stream_chunks}")
    print(f"stream total     : {res.stream_total_s:.3f}s")


if __name__ == "__main__":
    main()
