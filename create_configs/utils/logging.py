import os
from pathlib import Path


def new_version_dir(root_dir, version=0) -> str:
    while True:
        version_dir = os.path.join(root_dir, f"version_{version}")
        try:
            Path(version_dir).mkdir(parents=True, exist_ok=False)
            return version_dir
        except FileExistsError:
            version += 1
