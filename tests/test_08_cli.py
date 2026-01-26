import pytest


def test_cli_dry_run(capsys):
    import sys

    sys.path.append("src")
    from tts_ms import cli

    code = cli.main(["--text", "dry run test", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "DRY_RUN_OK" in out


@pytest.mark.slow
def test_cli_synth_outputs_wav():
    import shutil
    import sys
    from pathlib import Path
    from uuid import uuid4

    sys.path.append("src")
    from tts_ms import cli

    base_dir = Path("cli_test_outputs") / str(uuid4())
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / "out.wav"

    try:
        code = cli.main(["--text", "Merhaba.", "--out", str(out_path)])
        assert code == 0
        data = out_path.read_bytes()
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WAVE"
        assert len(data) > 44
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
