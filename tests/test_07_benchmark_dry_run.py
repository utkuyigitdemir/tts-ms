def test_benchmark_dry_run():
    import subprocess
    import sys
    p = subprocess.run(
        [sys.executable, "scripts/benchmark.py", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "BENCHMARK_DRY_RUN_OK" in p.stdout
