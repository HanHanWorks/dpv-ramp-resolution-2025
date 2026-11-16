"""Resolve restricted-data project root without changing the scientific analysis."""
import os
from pathlib import Path

def find_project_root():
    def valid(p):
        return ((p / "data/processed").is_dir() and
                (p / "paper_writing_package_seta_major_revision").is_dir())
    supplied = os.environ.get("SETA_PROJECT_ROOT")
    if supplied:
        root = Path(supplied).expanduser().resolve()
        if valid(root):
            return root
        raise FileNotFoundError("SETA_PROJECT_ROOT must contain data/processed and paper_writing_package_seta_major_revision")
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents, *Path(__file__).resolve().parents]
    for root in candidates:
        if valid(root):
            return root
    raise FileNotFoundError("Restricted raw inputs are not included. Set SETA_PROJECT_ROOT to the authorized source project containing data/processed and paper_writing_package_seta_major_revision.")
