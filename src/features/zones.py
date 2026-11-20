"""Fixed zone category list, shared by training, evaluation and serving.

The categories come from taxi_zone_lookup.csv (never from the training data), so a model
trained on Jan-Feb still maps every valid zone to the same category code, and a zone it
never saw (or an invalid ID) becomes NaN instead of crashing or silently shifting codes.
"""
from pathlib import Path

import pandas as pd


def load_zone_categories(lookup_path: str | Path) -> list[int]:
    """Sorted LocationIDs from the lookup CSV."""
    return sorted(int(z) for z in pd.read_csv(lookup_path)["LocationID"])


def apply_zone_categories(zones: pd.Series, categories: list[int]) -> pd.Series:
    """Cast to a categorical with the fixed category list; values outside it become NaN."""
    return pd.Series(pd.Categorical(zones, categories=categories), index=zones.index, name=zones.name)
