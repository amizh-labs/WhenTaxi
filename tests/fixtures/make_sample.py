"""Generate tests/fixtures/trips_clean_sample.parquet (cleaned-table contract).

Run: python tests/fixtures/make_sample.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "trips_clean_sample.parquet"


def make_sample(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    pickup = start + pd.to_timedelta(rng.integers(0, 31 * 24 * 60, n), unit="m")
    return pd.DataFrame(
        {
            "pickup_ts": pickup,
            "PULocationID": rng.integers(1, 20, n).astype("int64"),
            "DOLocationID": rng.integers(1, 20, n).astype("int64"),
            "duration_min": rng.uniform(1, 60, n),
            "month": "2026-01",
        }
    ).sort_values("pickup_ts", ignore_index=True)


if __name__ == "__main__":
    make_sample().to_parquet(OUT, index=False)
    print(f"wrote {OUT}")
