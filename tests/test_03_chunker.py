def test_normalize_and_chunk():
    import sys
    sys.path.append("src")

    from tts_ms.utils.text import normalize_tr
    from tts_ms.tts.chunker import chunk_text

    raw = "  Merhaba   Utku !  Bugün   mülakat...  Hazır  mısın?  Harika; başlayalım: şimdi.  "
    norm, t1 = normalize_tr(raw)

    assert "  " not in norm
    assert norm.startswith("Merhaba")
    assert isinstance(t1.get("normalize"), float)

    cr = chunk_text(norm, max_chars=40)
    assert len(cr.chunks) >= 3
    assert all(len(c) <= 40 for c in cr.chunks)
    assert isinstance(cr.timings_s.get("chunk"), float)
