from pathlib import Path


READ_RESULTS_PATH = Path(__file__).with_name("results") / "read_results.py"

exec(
    compile(READ_RESULTS_PATH.read_text(encoding="utf-8"), str(READ_RESULTS_PATH), "exec"),
    globals(),
)
