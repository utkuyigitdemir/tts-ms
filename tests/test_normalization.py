from tts_ms.utils.text import normalize_tr


def test_normalize_tr_numbers():
    text, timings = normalize_tr("Ben 12 yaşındayım ve 5 elma yedim.")
    assert text == "Ben on iki yaşındayım ve beş elma yedim."

def test_normalize_tr_abbreviations():
    text, timings = normalize_tr("Dr. Ahmet ve Prof. Mehmet geldi.")
    assert text == "Doktor Ahmet ve Profesör Mehmet geldi."

def test_normalize_tr_dates():
    text, timings = normalize_tr("Tarih 15.04.2023 idi.")
    assert text == "Tarih on beş dört 2023 idi."

def test_normalize_tr_combined():
    text, timings = normalize_tr("Dr. Ayşe 12.05.2024 tarihinde 3 hasta gördü.")
    assert text == "Doktor Ayşe on iki beş 2024 tarihinde üç hasta gördü."
