"""Feast definitions and store construction (code-only, no checked-in yaml).

Offline store: parquet snapshot file. Online store: local SQLite. Both free and local.
"""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureStore, FeatureView, Field, FileSource, RepoConfig
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.types import Float32
from feast.value_type import ValueType

from features.store.snapshots import ENTITY_KEY, FEATURE_NAMES

FEATURE_VIEW = "zone_pair_stats"
FEATURE_REFS = [f"{FEATURE_VIEW}:{name}" for name in FEATURE_NAMES]


def make_store(root: Path, snapshot_path: Path) -> tuple[FeatureStore, list]:
    """Create a FeatureStore rooted at ``root`` reading snapshots from ``snapshot_path``."""
    root.mkdir(parents=True, exist_ok=True)
    config = RepoConfig(
        project="eta",
        provider="local",
        registry=str(root / "registry.db"),
        online_store=SqliteOnlineStoreConfig(path=str(root / "online.db")),
        offline_store={"type": "file"},
        entity_key_serialization_version=3,
    )
    zone_pair = Entity(name=ENTITY_KEY, join_keys=[ENTITY_KEY], value_type=ValueType.STRING)
    source = FileSource(path=str(snapshot_path), timestamp_field="event_timestamp")
    view = FeatureView(
        name=FEATURE_VIEW,
        entities=[zone_pair],
        ttl=timedelta(0),  # no expiry: snapshots end with an explicit empty-window row
        schema=[Field(name=name, dtype=Float32) for name in FEATURE_NAMES],
        source=source,
    )
    return FeatureStore(config=config), [zone_pair, view]
