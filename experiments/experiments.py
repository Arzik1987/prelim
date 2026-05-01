from pathlib import Path


RUN_PATH = Path(__file__).with_name("run.py")

exec(compile(RUN_PATH.read_text(encoding="utf-8"), str(RUN_PATH), "exec"), globals())
