"""
Shared utility functions for project-wide logging and data I/O.

This module provides:
  - Simple console headings (`print_heading`, `print_sub_heading`)
  - A consistent way to resolve paths under the project `data/` directory
  - Helpers to load/save CSV + Parquet datasets
"""

from pathlib import Path
from typing import Final

import pandas as pd

# Visual dividers for console logging
SECTION_DIVIDER: Final[str] = "=" * 40
SUB_SECTION_DIVIDER: Final[str] = "-" * 40

# Project root = one level above src/
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"


def print_heading(title: str) -> None:
    """
    Print a visually distinct major section header to the console.

    Example
    -------
    >>> print_heading("Loading Dataset")
    """
    print(f"\n{SECTION_DIVIDER}\n{title}\n{SECTION_DIVIDER}\n")


def print_sub_heading(title: str) -> None:
    """
    Print a visually distinct sub-section header to the console.

    Example
    -------
    >>> print_sub_heading("Train / Validation Split")
    """
    print(f"\n{SUB_SECTION_DIVIDER}\n{title}\n{SUB_SECTION_DIVIDER}\n")


def data_path(name: str) -> Path:
    """
    Resolve a CSV path under the `data/` directory.

    Parameters
    ----------
    name : str
        Base name of the dataset (without `.csv`).

    Returns
    -------
    Path
        Full path to `<PROJECT_ROOT>/data/<name>.csv`.

    Example
    -------
    >>> data_path("Dataset")
    PosixPath('.../data/Dataset.csv')
    """
    return DATA_DIR / f"{name}.csv"


def load_data(name: str, prefer_parquet: bool = True) -> pd.DataFrame:
    """
    Load a dataset from `data/` given its base name (without `.csv`).

    By default this will try to read a Parquet file first (``<name>.parquet``)
    if it exists, and fall back to the CSV file (``<name>.csv``) otherwise.

    Parameters
    ----------
    name : str
        Base name of the dataset under `data/` (no extension).
    prefer_parquet : bool, default True
        If True, attempt to load ``<name>.parquet`` first and fall back to CSV.
        If False, always load from CSV.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.
    """
    csv_path = data_path(name)
    pq_path = csv_path.with_suffix(".parquet")

    print_heading("Loading Dataset")

    if prefer_parquet and pq_path.exists():
        print(f"Reading dataset from (Parquet): {pq_path}")
        df = pd.read_parquet(pq_path)
    else:
        # Either Parquet is not preferred or not present; use CSV.
        print(f"Reading dataset from (CSV): {csv_path}")
        df = pd.read_csv(csv_path)

    print(
        f"Dataset loaded successfully with "
        f"{len(df):,} rows and {len(df.columns)} columns."
    )
    return df


def save_data(
    df: pd.DataFrame,
    name: str,
    index: bool = False,
) -> None:
    """
    Save a dataframe to `data/` with the given base name (without `.csv`).

    This writes both:
      - `<name>.csv`
      - `<name>.parquet`

    Parameters
    ----------
    df : pandas.DataFrame
        Data to save.
    name : str
        Base name of the output files (no extension).
    index : bool, default False
        Whether to write the index to disk.
    """
    path = data_path(name)  # e.g. .../data/name.csv
    print_heading("Saving Dataset")
    print(f"Writing CSV dataset to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"CSV saved with {len(df):,} rows and {len(df.columns)} columns.")

    pq_path = path.with_suffix(".parquet")
    print(f"Writing Parquet dataset to: {pq_path}")
    df.to_parquet(pq_path, index=index)
    print("Parquet saved.")


def load_raw_data() -> pd.DataFrame:
    """
    Backwards-compatible helper to specifically load `data/Dataset.csv`.

    Returns
    -------
    pandas.DataFrame
        The original raw dataset used throughout the project.
    """
    return load_data("Dataset")