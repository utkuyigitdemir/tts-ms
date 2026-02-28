def test_cache_and_storage_roundtrip():
    import sys
    sys.path.append("src")

    from tts_ms.tts.cache import CacheItem, TinyLRUCache
    from tts_ms.tts.storage import hash_dict, make_key, save_wav, try_load_wav

    base_dir = "storage_test"
    text = "Merhaba"
    settings_hash = hash_dict({"engine": "legacy", "preset": "balanced"})
    key = make_key(
        text,
        "spk",
        "tr",
        engine_type="legacy",
        model_id="xtts",
        settings_hash=settings_hash,
    )
    other_key = make_key(
        text,
        "spk",
        "tr",
        engine_type="piper",
        model_id="piper",
        settings_hash=settings_hash,
    )
    assert key != other_key
    wav = b"RIFF....WAVE" + b"x" * 100  # fake but ok for storage test

    # storage write/read
    save_wav(base_dir, key, wav)
    loaded, _ = try_load_wav(base_dir, key)
    assert loaded == wav

    # cache set/get
    c = TinyLRUCache(max_items=2)
    c.set(key, CacheItem(wav_bytes=wav, sample_rate=24000))
    item, _ = c.get(key)
    assert item is not None
    assert item.wav_bytes == wav
