# TTS-MS Benchmark Examples

This document presents real-world benchmark results from TTS-MS engines, analyzing performance, resource usage, and efficiency across different synthesis scenarios.

**Audio Samples:** [Listen to generated audio files on OneDrive](https://innovacomtr-my.sharepoint.com/:f:/g/personal/udemir_innova_com_tr/IgAnllV_7lDiQp4Euj_R4WSIAYqyZBrHslTaup9Wqor0CkQ?e=exqlGp)

---

## Table of Contents

1. [Benchmark Overview](#1-benchmark-overview)
2. [Piper Engine Results (Colab)](#2-piper-engine-results)
3. [Legacy XTTS v2 Results (Colab)](#3-legacy-xtts-v2-results)
4. [Comparative Analysis (Piper vs XTTS v2)](#4-comparative-analysis)
5. [Resource Efficiency Analysis](#5-resource-efficiency-analysis)
6. [All 9 Engines - CPU Benchmarks](#additional-engine-results-cpu-benchmarks)

---

## 1. Benchmark Overview

### Test Scenario

The benchmark simulates a **Turkish job interview** scenario with 5 question-answer pairs. This tests both short utterances (questions) and longer responses (answers), providing a realistic use case for conversational TTS.

### Test Questions (Turkish)

| # | Question | Answer Summary |
|---|----------|----------------|
| 1 | Sizi neden işe almalıyız? | Analitik düşünme, takım çalışması, problem çözme |
| 2 | Siz bizi neden seçtiniz? | Şirketin yenilikçi yaklaşımı, kariyer hedefleri |
| 3 | Kötü özellikleriniz nelerdir? | Aşırı detaycılık, zaman yönetimi farkındalığı |
| 4 | Beş yıl sonra kendinizi nerede görüyorsunuz? | Teknik liderlik, ekip yönetimi |
| 5 | Maaş beklentiniz nedir? | Piyasa koşulları, esneklik |

### Text Lengths

- **Questions:** 23-44 characters (short utterances)
- **Answers:** 150-240 characters (paragraph-length responses)

---

## 2. Piper Engine Results

### Environment

| Parameter | Value |
|-----------|-------|
| **Platform** | Google Colab |
| **CPU** | 2 cores |
| **GPU** | N/A (CPU-only) |
| **Model** | tr_TR-dfki-medium |
| **Date** | 2026-01-25 |

### Performance Summary

| Metric | Value |
|--------|-------|
| **Initialization** | 0.04s |
| **Warmup** | 4.75s |
| **Avg Question Time** | 0.29s |
| **Avg Answer Time** | 1.51s |
| **Total Audio Size** | 3,056 KB |
| **Success Rate** | 10/10 (100%) |

### Resource Usage

| Resource | Average | Maximum |
|----------|---------|---------|
| **CPU** | 48% | 50% |
| **RAM Delta** | - | +181.7 MB |
| **GPU** | N/A | N/A |

### Detailed Results

| Sample | Time | Size | CPU | Status |
|--------|------|------|-----|--------|
| 01_soru | 0.23s | 85.5 KB | 48% | OK |
| 01_cevap | 1.61s | 647.0 KB | 50% | OK |
| 02_soru | 0.27s | 82.5 KB | 48% | OK |
| 02_cevap | 1.41s | 564.5 KB | 50% | OK |
| 03_soru | 0.28s | 85.5 KB | 49% | OK |
| 03_cevap | 1.44s | 481.0 KB | 49% | OK |
| 04_soru | 0.39s | 122.0 KB | 48% | OK |
| 04_cevap | 1.34s | 392.0 KB | 44% | OK |
| 05_soru | 0.26s | 77.5 KB | 48% | OK |
| 05_cevap | 1.73s | 518.5 KB | 45% | OK |

### Resource Log (Step-by-Step)

```
[INIT]     0.04s | RAM: +2.3 MB
[WARMUP]   4.75s | CPU: 11% | RAM: +141.8 MB
[01_SORU]  0.23s | 85.5 KB | CPU: 48% | RAM: +22.8 MB
[01_CEVAP] 1.61s | 647.0 KB | CPU: 50% | RAM: +136.5 MB
[02_SORU]  0.27s | 82.5 KB | CPU: 48% | RAM: +4.7 MB
[02_CEVAP] 1.41s | 564.5 KB | CPU: 50% | RAM: +1.2 MB
[03_SORU]  0.28s | 85.5 KB | CPU: 49% | RAM: +0.0 MB
[03_CEVAP] 1.44s | 481.0 KB | CPU: 49% | RAM: +0.0 MB
[04_SORU]  0.39s | 122.0 KB | CPU: 48% | RAM: +0.0 MB
[04_CEVAP] 1.34s | 392.0 KB | CPU: 44% | RAM: +2.4 MB
[05_SORU]  0.26s | 77.5 KB | CPU: 48% | RAM: +0.0 MB
[05_CEVAP] 1.73s | 518.5 KB | CPU: 45% | RAM: +14.1 MB
```

### Piper Efficiency Metrics

| Metric | Value |
|--------|-------|
| **Characters per Second** | ~85 chars/s |
| **Audio KB per Second** | ~340 KB/s |
| **Real-time Factor (RTF)** | ~0.15x (6.7x faster than real-time) |

---

## 3. Legacy XTTS v2 Results

### Environment

| Parameter | Value |
|-----------|-------|
| **Platform** | Google Colab (Miniconda Python 3.11) |
| **GPU** | NVIDIA A100-SXM4-40GB |
| **Model** | xtts_v2 |
| **Date** | 2026-01-26 |

### Performance Summary

| Metric | Value |
|--------|-------|
| **Initialization** | 0.03s |
| **Warmup** | 72.17s |
| **Avg Question Time** | 0.91s |
| **Avg Answer Time** | 6.08s |
| **Total Audio Size** | 4,022 KB |
| **Success Rate** | 10/10 (100%) |

### Resource Usage

| Resource | Average | Maximum |
|----------|---------|---------|
| **CPU** | 8% | 9% |
| **RAM Delta** | - | +6.1 MB |
| **GPU Utilization** | 32% | 42% |
| **VRAM** | - | 3,353 MB |

### Detailed Results

| Sample | Time | Size | CPU | GPU | VRAM | Status |
|--------|------|------|-----|-----|------|--------|
| 01_soru | 0.75s | 97.6 KB | 9% | 27% | 2,887 MB | OK |
| 01_cevap | 7.40s | 830.6 KB | 8% | 42% | 3,293 MB | OK |
| 02_soru | 0.86s | 104.1 KB | 9% | 26% | 3,293 MB | OK |
| 02_cevap | 5.87s | 674.6 KB | 8% | 34% | 3,293 MB | OK |
| 03_soru | 0.94s | 115.1 KB | 9% | 28% | 3,293 MB | OK |
| 03_cevap | 5.74s | 650.1 KB | 8% | 41% | 3,293 MB | OK |
| 04_soru | 1.17s | 139.1 KB | 9% | 29% | 3,293 MB | OK |
| 04_cevap | 5.04s | 582.6 KB | 8% | 40% | 3,293 MB | OK |
| 05_soru | 0.80s | 104.1 KB | 9% | 27% | 3,293 MB | OK |
| 05_cevap | 6.33s | 724.1 KB | 8% | 27% | 3,353 MB | OK |

### Resource Log (Step-by-Step)

```
[INIT]     0.03s | RAM: +4.1 MB | GPU: 0%
[WARMUP]   72.17s | RAM: +1602.1 MB | GPU: 26%
[01_SORU]  0.75s | 97.6 KB | CPU: 9% | GPU: 27% | VRAM: 2887 MB
[01_CEVAP] 7.40s | 830.6 KB | CPU: 8% | GPU: 42% | VRAM: 3293 MB
[02_SORU]  0.86s | 104.1 KB | CPU: 9% | GPU: 26% | VRAM: 3293 MB
[02_CEVAP] 5.87s | 674.6 KB | CPU: 8% | GPU: 34% | VRAM: 3293 MB
[03_SORU]  0.94s | 115.1 KB | CPU: 9% | GPU: 28% | VRAM: 3293 MB
[03_CEVAP] 5.74s | 650.1 KB | CPU: 8% | GPU: 41% | VRAM: 3293 MB
[04_SORU]  1.17s | 139.1 KB | CPU: 9% | GPU: 29% | VRAM: 3293 MB
[04_CEVAP] 5.04s | 582.6 KB | CPU: 8% | GPU: 40% | VRAM: 3293 MB
[05_SORU]  0.80s | 104.1 KB | CPU: 9% | GPU: 27% | VRAM: 3293 MB
[05_CEVAP] 6.33s | 724.1 KB | CPU: 8% | GPU: 27% | VRAM: 3353 MB
```

### XTTS v2 Efficiency Metrics

| Metric | Value |
|--------|-------|
| **Characters per Second** | ~25 chars/s |
| **Audio KB per Second** | ~115 KB/s |
| **Real-time Factor (RTF)** | ~0.5x (2x faster than real-time) |
| **VRAM Efficiency** | ~1.2 KB audio per MB VRAM |

---

## 4. Comparative Analysis

### Head-to-Head Comparison

| Metric | Piper | Legacy XTTS v2 | Winner |
|--------|-------|----------------|--------|
| **Warmup Time** | 4.75s | 72.17s | Piper (15x faster) |
| **Avg Question Synthesis** | 0.29s | 0.91s | Piper (3x faster) |
| **Avg Answer Synthesis** | 1.51s | 6.08s | Piper (4x faster) |
| **Total Synthesis Time** | 9.0s | 34.9s | Piper (3.9x faster) |
| **Total Audio Size** | 3,056 KB | 4,022 KB | XTTS (32% larger) |
| **Audio Quality** | Good | Excellent | XTTS v2 |
| **Voice Cloning** | No | Yes | XTTS v2 |
| **GPU Required** | No | Yes | Piper |
| **CPU Usage** | 48% avg | 8% avg | XTTS v2 (6x lower) |
| **Memory Footprint** | 182 MB | 3,353 MB VRAM | Piper |

### Synthesis Speed Comparison

```
Question Synthesis (avg):
Piper:    ████████████████████████████████████████ 0.29s
XTTS v2:  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 0.91s

Answer Synthesis (avg):
Piper:    ████████████████████████████████████████ 1.51s
XTTS v2:  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 6.08s
```

### Audio Output Size Comparison

```
Total Audio Generated:
Piper:    ████████████████████████████████████████████████████████████████████████████ 3,056 KB
XTTS v2:  ████████████████████████████████████████████████████████████████████████████████████████████████████ 4,022 KB
```

The larger audio size for XTTS v2 indicates higher sample rate (24kHz vs 22kHz) and potentially richer audio content.

---

## 5. Resource Efficiency Analysis

### CPU Efficiency

| Engine | CPU Usage | Interpretation |
|--------|-----------|----------------|
| **Piper** | 48% avg (2 cores) | Efficiently uses available CPU cores |
| **XTTS v2** | 8% avg | Minimal CPU load, GPU-bound workload |

**Analysis:** Piper fully utilizes CPU resources (48% on 2-core = ~96% single-thread equivalent), making it ideal for CPU-only deployments. XTTS v2 offloads computation to GPU, leaving CPU nearly idle.

### Memory Efficiency

| Engine | Memory Type | Usage | Efficiency |
|--------|-------------|-------|------------|
| **Piper** | RAM | +182 MB | Excellent - minimal footprint |
| **XTTS v2** | VRAM | 3,353 MB | Moderate - requires dedicated GPU |
| **XTTS v2** | RAM | +1,602 MB (warmup) | Model loading overhead |

**Analysis:** Piper's 182 MB RAM footprint allows deployment on resource-constrained environments. XTTS v2 requires ~3.4 GB VRAM, limiting it to systems with dedicated GPUs.

### GPU Utilization (XTTS v2)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Avg GPU Util** | 32% | Room for concurrent requests |
| **Max GPU Util** | 42% | Peak during long synthesis |
| **VRAM Stable** | 3,293 MB | Consistent after warmup |

**Analysis:** XTTS v2 uses only 32-42% GPU utilization on A100, suggesting potential for concurrent request handling (up to 2-3 simultaneous requests before saturation).

### Throughput Analysis

| Metric | Piper | XTTS v2 | Ratio |
|--------|-------|---------|-------|
| **Chars/Second** | 85 | 25 | 3.4x |
| **KB Audio/Second** | 340 | 115 | 3.0x |
| **Samples/Minute** | ~60 | ~15 | 4.0x |

**Analysis:** Piper delivers 3-4x higher throughput, making it suitable for high-volume, low-latency applications. XTTS v2 trades speed for quality.

### Cost Efficiency (Estimated)

| Scenario | Piper | XTTS v2 |
|----------|-------|---------|
| **Hardware** | Any CPU | GPU required (~$1-3/hr cloud) |
| **Synthesis Cost** | ~$0.001/min audio | ~$0.01/min audio |
| **Concurrent Capacity** | 1 per CPU core | 2-3 per GPU |

---

## Appendix A: Colab Benchmark Data

The original Piper and XTTS v2 benchmarks above were run on Google Colab. Raw data files:

```
output/
├── piper_benchmark/
│   ├── summary.txt      # Human-readable summary
│   ├── results.json     # Machine-readable results
│   └── audio/           # Generated audio files
└── legacy_benchmark/
    ├── summary.txt
    ├── results.json
    └── audio/
```

---

## Additional Engine Results (CPU Benchmarks)

All 9 engines were benchmarked on a 22-core CPU machine (no GPU) on 2026-02-26. The same 5 Turkish interview question-answer pairs were used where supported; English-only engines ran 10 English tests.

### All Engines Summary

| Engine | Startup | Tests | Pass | Avg Time | Total Time | Audio Output | Turkish |
|--------|---------|-------|------|----------|------------|-------------|---------|
| **Piper** | 3.8s | 20 | 20 | 0.27s | 5.3s | 130.6s playback | Yes |
| **Kokoro** | 3.1s | 10 | 10 | 1.85s | 18.5s | 68.8s playback | No |
| **StyleTTS2** | 106.3s | 10 | 10 | 8.43s | 84.3s | 79.8s playback | No |
| **Legacy XTTS v2** | 188.4s | 20 | 20 | 12.04s | 240.7s | 150.4s playback | Yes |
| **Chatterbox** | 261.4s | 20 | 20 | 32.46s | 649.2s | 119.8s playback | Yes |
| **CosyVoice** | 54.0s | 10 | 10 | 53.97s | 539.7s | 67.4s playback | No |
| **F5-TTS** | 351.2s | 10 | 10 | 92.11s | 921.1s | 67.1s playback | No |
| **VibeVoice** | 450.9s | 10 | 10 | 198.07s | 1980.7s | 112.1s playback | No |
| **Qwen3-TTS** | 194.5s | 10 | 3 | 320.87s | 3208.7s | 7.0s playback | No |

> **Note:** CPU percentages are raw psutil values (0-2200% range for 22 cores). Engines marked "No" for Turkish ran English-only tests. Qwen3-TTS had 7 timeout failures on CPU — it is designed for GPU.

### Resource Usage Comparison

| Engine | Avg CPU | Peak CPU | Avg RAM Delta | Peak RAM Delta |
|--------|---------|----------|---------------|----------------|
| **Piper** | 836.9% | 874.6% | +16.4 MB | +129.8 MB |
| **Kokoro** | 865.4% | 876.9% | +0.0 MB | +0.0 MB |
| **StyleTTS2** | 501.7% | 566.9% | +109.2 MB | +597.1 MB |
| **Legacy XTTS v2** | 590.2% | 600.2% | +21.0 MB | +161.2 MB |
| **Chatterbox** | 594.4% | 610.8% | +31.4 MB | +345.8 MB |
| **CosyVoice** | 606.9% | 613.6% | -9.8 MB | +78.2 MB |
| **F5-TTS** | 579.3% | 583.0% | +9.5 MB | +132.7 MB |
| **VibeVoice** | 502.5% | 535.5% | +0.6 MB | +1.9 MB |
| **Qwen3-TTS** | 202.3% | 233.4% | -403.1 MB | +198.9 MB |

### Per-Engine Notes

**Kokoro** — CPU-only ONNX engine. Fast startup (3.1s), good throughput (~37 chars/s). English-only, no Turkish. Preset voices only (af_sarah, am_adam, etc.).

**StyleTTS2** — English-only with style transfer. Moderate speed on CPU (8.43s avg). Highest RAM delta (+597 MB peak) due to model architecture.

**Chatterbox** — Expressive speech with paralinguistics. Supports Turkish natively. Slower startup (261s) but acceptable synthesis time (32s avg on CPU).

**CosyVoice** — Natural prosody engine. English-only in current test setup. Moderate CPU performance (54s avg). Requires Python 3.10.

**F5-TTS** — Voice cloning model. Slowest-to-start engine (351s). Very slow on CPU (92s avg) — designed for GPU. English-only (requires Turkish reference audio for TR).

**VibeVoice** — Microsoft research model. Extremely slow on CPU (198s avg, 451s startup). Peak RAM only +1.9 MB (efficient memory management). English-only in tests.

**Qwen3-TTS** — Alibaba model. Only 3/10 tests passed on CPU (7 timed out). CPU is insufficient for this engine — GPU is required for practical use.

---

## 6. Recommendations

(Updated with all 9 engines)

### Deployment Recommendations

| Use Case | Recommended Engine | Reasoning |
|----------|-------------------|-----------|
| IVR/Call Center | Piper | High throughput, consistent latency, CPU-only |
| Virtual Assistant | XTTS v2 or Chatterbox | Natural conversation quality with Turkish |
| Audiobook Generation | XTTS v2 or VibeVoice | Extended listening requires quality (GPU) |
| Real-time Streaming | Piper or Kokoro | Low latency, predictable performance |
| Voice Cloning | F5-TTS, CosyVoice, Chatterbox, XTTS v2, Qwen3-TTS, or VibeVoice | 6 engines support cloning |
| Edge Deployment | Piper or Kokoro | CPU-only, small footprint |
| Style/Expressiveness | StyleTTS2 or Chatterbox | Style transfer and paralinguistics |
| Multilingual (non-TR) | CosyVoice or Qwen3-TTS | Broad language support |

---

## Appendix: Raw Data Files

The complete benchmark data is available in the `results/` directory:

```
results/
├── summary.txt           # Multi-engine summary
├── piper/                # Per-engine detailed results
├── kokoro/
├── legacy/
├── f5tts/
├── styletts2/
├── chatterbox/
├── cosyvoice/
├── qwen3tts/
└── vibevoice/
```

---

**Audio Samples:** [Listen to all generated audio files](https://innovacomtr-my.sharepoint.com/:f:/g/personal/udemir_innova_com_tr/IgAnllV_7lDiQp4Euj_R4WSIAYqyZBrHslTaup9Wqor0CkQ?e=exqlGp)

---

*Last Updated: February 2026*
