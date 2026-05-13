# PRELIM Code Reorganization Suggestions

Temporary note from the repository inspection. Completed items have been removed from this list.

## Remaining follow-ups

- Replace string-formatted CSV result writes with a small dataclass plus `csv.DictWriter`.
- Add a cleanup helper for generated local artifacts such as `__pycache__` and egg-info directories.
