
import os
import sys
import json
import time
import psutil
from datetime import datetime

# Fix matplotlib backend issue
os.environ["MPLBACKEND"] = "agg"

# Environment setup
os.environ["TTS_MODEL_TYPE"] = "legacy"
os.environ["TTS_MS_LOG_LEVEL"] = "4"
os.environ["TTS_MS_RESOURCES_ENABLED"] = "1"
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TTS_HOME"] = r"C:\Users\udemir\Desktop\projects\tts-ms\notebooks\cache\xtts"

ENGINE_NAME = "legacy"
MODEL_NAME = "xtts_v2"
BASE_DIR = r"C:\Users\udemir\Desktop\projects\tts-ms\notebooks\output\legacy"
AUDIO_DIR = r"C:\Users\udemir\Desktop\projects\tts-ms\notebooks\output\legacy\audio"

from tts_ms.services import TTSService
from tts_ms.services.tts_service import SynthesizeRequest
from tts_ms.core.config import Settings

process = psutil.Process()
CPU_COUNT = psutil.cpu_count()
resource_logs = []
results = []

def get_resources():
    return {"cpu": process.cpu_percent(), "ram_mb": process.memory_info().rss / 1024 / 1024}

QUESTIONS = [
    {"id": "01", "question": "Sizi neden işe almalıyız?",
     "answer": "Güçlü analitik düşünme becerilerim ve takım çalışmasına yatkınlığım sayesinde projelere değer katabilirim. Ayrıca sürekli öğrenmeye açık yapım ve problem çözme yeteneklerim, şirketinizin hedeflerine ulaşmasında önemli katkılar sağlayacaktır."},
    {"id": "02", "question": "Siz bizi neden seçtiniz?",
     "answer": "Şirketinizin yenilikçi yaklaşımı ve sektördeki lider konumu beni çok etkiledi. Kariyer hedeflerimle örtüşen bu ortamda kendimi geliştirebileceğime ve anlamlı projeler üzerinde çalışabileceğime inanıyorum."},
    {"id": "03", "question": "Kötü özellikleriniz nelerdir?",
     "answer": "Bazen aşırı detaycı olabiliyorum, bu da zaman yönetimimi olumsuz etkileyebiliyor. Ancak bu özelliğimin farkındayım ve önceliklendirme teknikleri kullanarak bu durumu yönetmeye çalışıyorum."},
    {"id": "04", "question": "Beş yıl sonra kendinizi nerede görüyorsunuz?",
     "answer": "Beş yıl içinde teknik liderlik pozisyonunda olmayı hedefliyorum. Ekip yönetimi deneyimi kazanarak şirketin büyümesine stratejik katkılar sağlamak istiyorum."},
    {"id": "05", "question": "Maaş beklentiniz nedir?",
     "answer": "Piyasa koşullarını ve pozisyonun gerekliliklerini değerlendirerek, deneyimime ve yeteneklerime uygun rekabetçi bir maaş beklentim var. Bu konuda esnek olmaya ve karşılıklı bir anlaşmaya varmaya açığım."}
]

print("="*60)
print("INITIALIZING TTS SERVICE")
print("="*60)

res_before = get_resources()
start = time.time()

settings = Settings(raw={
    "tts": {
        "engine": "legacy",
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "device": "cuda",
        "default_language": "tr",
        "default_speaker": "Ana Florence"
    },
    "cache": {"enabled": False},
    "storage": {"enabled": False},
    "logging": {"level": 4}
})
service = TTSService(settings)
init_time = time.time() - start
res_after = get_resources()
resource_logs.append({"stage": "init", "duration": init_time, "cpu": res_after["cpu"], "ram_delta": res_after["ram_mb"] - res_before["ram_mb"]})
print(f"Initialized in {init_time:.2f}s")

# Warmup
print("\nWarmup...")
res_before = get_resources()
start = time.time()
_ = service.synthesize(SynthesizeRequest(text="Merhaba."), request_id="warmup")
warmup_time = time.time() - start
res_after = get_resources()
resource_logs.append({"stage": "warmup", "duration": warmup_time, "cpu": res_after["cpu"], "ram_delta": res_after["ram_mb"] - res_before["ram_mb"]})
print(f"Warmup done in {warmup_time:.2f}s")

print("\n" + "="*60)
print(f"SYNTHESIZING {len(QUESTIONS)} QUESTIONS + ANSWERS")
print("="*60)

for q in QUESTIONS:
    print(f"\n[{q['id']}] {q['question']}")

    for typ, text in [("soru", q["question"]), ("cevap", q["answer"])]:
        print(f"  {typ.upper()}: ", end="", flush=True)
        res_before = get_resources()
        start = time.time()
        try:
            result = service.synthesize(SynthesizeRequest(text=text), request_id=f"{q['id']}_{typ}")
            elapsed = time.time() - start
            res_after = get_resources()

            path = os.path.join(AUDIO_DIR, f"{q['id']}_{typ}.wav")
            with open(path, "wb") as f:
                f.write(result.wav_bytes)

            cpu_norm = res_after["cpu"] / CPU_COUNT
            ram_delta = res_after["ram_mb"] - res_before["ram_mb"]

            resource_logs.append({"stage": f"{q['id']}_{typ}", "duration": elapsed, "cpu": res_after["cpu"], "cpu_norm": cpu_norm, "ram_delta": ram_delta, "size_kb": len(result.wav_bytes)/1024})
            results.append({"id": q["id"], "type": typ, "time": elapsed, "size": len(result.wav_bytes), "cpu": cpu_norm, "ram_delta": ram_delta, "status": "OK"})

            print(f"{elapsed:.2f}s | {len(result.wav_bytes)/1024:.1f} KB | CPU:{cpu_norm:.0f}% | OK")
        except Exception as e:
            results.append({"id": q["id"], "type": typ, "time": time.time()-start, "size": 0, "cpu": 0, "ram_delta": 0, "status": f"FAIL: {e}"})
            print(f"FAIL: {e}")

successful = [r for r in results if r["status"] == "OK"]
print(f"\n" + "="*60)
print(f"COMPLETE: {len(successful)}/{len(results)} successful")
print("="*60)

# Save results
output_data = {
    "engine": ENGINE_NAME,
    "model": MODEL_NAME,
    "init_time": init_time,
    "warmup_time": warmup_time,
    "results": results,
    "resource_logs": resource_logs,
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(BASE_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {BASE_DIR}")
