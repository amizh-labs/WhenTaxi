"""The serving image ships the zone-centroid CSV and does not carry the one-time geo tooling (pyshp, pyproj)."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
CODE_LINES = [line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#")]  # comments may name libs
INSTRUCTIONS = chr(10).join(CODE_LINES)
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
GEO_LIBS = ("pyshp", "pyproj", "shapefile", "shapely", "geopandas")


def test_image_copies_the_committed_centroid_csv():
    assert (ROOT / "data" / "reference" / "zone_centroids.csv").exists()
    assert re.search(r"^COPY\s+data/reference\s+\./data/reference\s*$", DOCKERFILE, re.MULTILINE)
    assert "ETA_CENTROIDS_PATH=/app/data/reference/zone_centroids.csv" in DOCKERFILE


def test_dockerignore_does_not_exclude_the_reference_folder():
    # `data` (DVC-tracked raw files) is excluded; the re-include for data/reference must come after it
    lines = [line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()]
    assert "data" in lines and "!data/reference" in lines
    assert lines.index("!data/reference") > lines.index("data")


def test_image_installs_only_the_serving_extra_and_no_geo_libraries():
    installs = re.findall(r"pip install[^\n]*", DOCKERFILE)
    assert installs and all('".[serving]"' in line for line in installs)
    assert not any(re.search(r"\b" + lib + r"\b", INSTRUCTIONS.lower()) for lib in GEO_LIBS)


def test_geo_libraries_only_in_the_optional_geo_extra():
    project = PYPROJECT["project"]
    extras = project["optional-dependencies"]
    assert set(extras["geo"]) == {"pyshp", "pyproj"}
    for name, deps in [("dependencies", project["dependencies"]), *((k, v) for k, v in extras.items() if k != "geo")]:
        assert not [d for d in deps if d.lower().startswith(GEO_LIBS)], f"geo library leaked into {name}"
