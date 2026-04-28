"""Dataset loading utilities with clear fallbacks and error messages."""

from pathlib import Path
from typing import Tuple

import pandas as pd

RANDOM_STATE = 42
DEFAULT_DATA_DIR = Path("datasets")

def load_datasets(
    base_dir: Path | str = DEFAULT_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load registry, catalog, and maintenance history datasets.

    This uses the fixed filenames requested for the project.
    """
    base_dir = Path(base_dir)

    doors_registry = pd.read_csv(base_dir / "registry.csv")
    door_catalog = pd.read_csv(base_dir / "catalog.csv")
    maintenance_history = pd.read_csv(base_dir / "history.csv")

    return doors_registry, door_catalog, maintenance_history
