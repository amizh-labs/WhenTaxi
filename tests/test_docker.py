"""The serving image ships the zone-centroid CSV and does not carry the one-time geo tooling (pyshp, pyproj)."""
import re
import shlex
import tomllib
from pathlib import Path

import pytest

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


# ---- formatting: a hand edit once left a literal backslash-n inside an ENV instruction -----------------------------

BACKSLASH = chr(92)
BACKSLASH_N = BACKSLASH + "n"
NAME_VALUE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S+$")


def env_problems(text: str) -> list[str]:
    """Problems in the ENV instructions: bad continuations, or any token that is not a name=value pair."""
    problems, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if not lines[i].startswith("ENV "):
            i += 1
            continue
        j = i
        while lines[j].endswith(BACKSLASH) and j + 1 < len(lines):
            j += 1
            if not lines[j].startswith((" ", "\t")):
                problems.append(f"continuation line is not indented: {lines[j]!r}")
        if lines[j].endswith(BACKSLASH):
            problems.append("ENV ends with a dangling backslash")
        for line in lines[i : j + 1]:
            if re.search(r"\\\s+$", line):
                problems.append(f"whitespace after the continuation backslash: {line!r}")
        body = " ".join(line.removesuffix(BACKSLASH).strip() for line in lines[i : j + 1]).removeprefix("ENV ")
        try:
            problems += [f"not a name=value pair: {tok!r}" for tok in shlex.split(body) if not NAME_VALUE.match(tok)]
        except ValueError as exc:  # e.g. a stray backslash at the end
            problems.append(f"cannot parse the ENV values: {exc}")
        i = j + 1
    return problems


def test_dockerfile_has_no_literal_backslash_n():
    assert BACKSLASH_N not in DOCKERFILE


def test_every_env_instruction_is_name_value_pairs_with_proper_continuations():
    assert env_problems(DOCKERFILE) == []
    envs = {tok.split("=")[0] for tok in shlex.split(DOCKERFILE.replace(BACKSLASH + chr(10), " ")) if "=" in tok}
    assert {"ETA_CENTROIDS_PATH", "ETA_WEATHER_PATH", "MLFLOW_TRACKING_URI"} <= envs


@pytest.mark.parametrize("bad", [
    "ENV A=1 " + BACKSLASH_N + "    B=2",  # the original bug: a literal backslash-n in place of a line break
    "ENV A=1 " + BACKSLASH + chr(10) + "B=2",  # continuation line not indented
    "ENV A=1 " + BACKSLASH,  # dangling continuation
    "ENV A=1 " + BACKSLASH + " " + chr(10) + "    B=2",  # trailing space after the backslash
    "ENV A 1",  # legacy form, not name=value
])
def test_the_env_checker_catches_broken_instructions(bad):
    assert env_problems(bad)


def test_the_env_checker_accepts_a_well_formed_instruction():
    good = "ENV A=1 " + BACKSLASH + chr(10) + "    B=http://x:5000"
    assert env_problems(good) == []
