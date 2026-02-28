import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from tts_ms.utils.text import normalize_tr
# disable the module logger to prevent terminal garbling
import tts_ms.core.logging as core_log
core_log.get_logger("tts-ms.text").setLevel(logging.CRITICAL)

cases = [
    ("Ben 12 yaşındayım ve 5 elma yedim.", "Ben on iki yaşındayım ve beş elma yedim."),
    ("Dr. Ahmet ve Prof. Mehmet geldi.", "Doktor Ahmet ve Profesör Mehmet geldi."),
    ("Tarih 15.04.2023 idi.", "Tarih 15 04 2023 idi."),
    ("Dr. Ayşe 12.05.2024 tarihinde 3 hasta gördü.", "Doktor Ayşe 12 05 2024 tarihinde üç hasta gördü.")
]

with open("test_out.txt", "w", encoding="utf-8") as f:
    for i, (inp, expected) in enumerate(cases):
        act, timings = normalize_tr(inp)
        if act != expected:
            f.write(f"FAILED CASE {i+1}:\n")
            f.write(f"EXP: '{expected}'\n")
            f.write(f"ACT: '{act}'\n")
            f.write("-" * 20 + "\n")
        else:
            f.write(f"PASSED CASE {i+1}\n")
