from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "experiments"
    if __spec__ is not None and __spec__.parent != __package__:
        __spec__ = None

from .results.read_results import *  # noqa: F401,F403
from .results.read_results import main


if __name__ == "__main__":
    main()
