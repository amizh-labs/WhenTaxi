"""UI tests: streamlit.testing.v1.AppTest with requests.post mocked. No network, no API."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
ZONES = pd.read_csv(Path(__file__).resolve().parents[1] / "zones.csv").set_index("LocationID")
OK = {"eta_minutes": 27.46, "model_version": "7", "weather_source": "archive"}


def response(status=200, body=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.json.return_value = body if body is not None else {}
    return r


def new_app():
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def pick(box, zone_id):
    box.set_value(zone_id)  # select()/select_index() cannot cope with format_func, set_value takes the raw id


def press(at):
    at.sidebar.button[0].click().run()
    assert not at.exception
    return at


def test_page_loads_with_the_replay_note():
    at = new_app()
    assert at.title[0].value == "NYC taxi trip ETA"
    assert any("replay demo, not a live ETA service" in i.value for i in at.info)
    assert [b.label for b in at.sidebar.selectbox] == ["Pickup zone", "Dropoff zone"]
    assert len(at.metric) == 0 and len(at.error) == 0  # nothing predicted yet


def test_a_successful_prediction_shows_the_metrics_and_sends_only_three_fields(ui_app):
    at = new_app()
    pick(at.sidebar.selectbox[0], 161)
    pick(at.sidebar.selectbox[1], 132)
    with patch("requests.post", return_value=response(200, OK)) as post:
        press(at)
    (url,), kwargs = post.call_args
    assert url == "http://localhost:8000/predict"
    assert kwargs["json"] == {"pickup_zone": 161, "dropoff_zone": 132, "pickup_ts": "2026-03-15T18:00:00"}  # no distance
    assert kwargs["timeout"] > 0
    metrics = {m.label: m.value for m in at.metric}
    km = ui_app.haversine_km(*ZONES.loc[161, ["lat", "lon"]], *ZONES.loc[132, ["lat", "lon"]])
    assert metrics == {"ETA (minutes)": "27.5", "Model version": "7", "Weather source": "archive",
                       "Distance (km)": f"{km:.2f}"}
    assert 15 < km < 25  # Midtown to JFK
    assert len(at.error) == 0
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 1  # the session history


def test_history_grows_with_each_prediction():
    at = new_app()
    with patch("requests.post", return_value=response(200, OK)):
        press(at)
        press(at)
    assert len(at.dataframe[0].value) == 2


def test_api_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("API_URL", "http://api:8000/")
    at = new_app()
    with patch("requests.post", return_value=response(200, OK)) as post:
        press(at)
    assert post.call_args[0][0] == "http://api:8000/predict"


def test_a_422_shows_a_clear_error_and_no_metrics():
    body = {"detail": [{"loc": ["body", "pickup_zone"], "msg": "Input should be greater than or equal to 1"}]}
    at = new_app()
    with patch("requests.post", return_value=response(422, body)):
        press(at)
    assert len(at.error) == 1
    assert "HTTP 422" in at.error[0].value and "pickup_zone" in at.error[0].value
    assert len(at.metric) == 0


def test_a_timeout_shows_a_clear_error():
    at = new_app()
    with patch("requests.post", side_effect=requests.Timeout()):
        press(at)
    assert "did not answer" in at.error[0].value and len(at.metric) == 0


def test_a_connection_failure_names_the_api_url():
    at = new_app()
    with patch("requests.post", side_effect=requests.ConnectionError()):
        press(at)
    assert "Could not reach the API at http://localhost:8000" in at.error[0].value


def test_a_server_error_is_reported_as_the_apis_problem():
    at = new_app()
    with patch("requests.post", return_value=response(500, {"detail": "internal error"})):
        press(at)
    assert "HTTP 500" in at.error[0].value and "internal error" in at.error[0].value


def test_a_failure_clears_the_previous_result():
    at = new_app()
    with patch("requests.post", return_value=response(200, OK)):
        press(at)
    assert len(at.metric) == 4
    with patch("requests.post", side_effect=requests.Timeout()):
        press(at)
    assert len(at.metric) == 0 and len(at.error) == 1


def test_excluded_zones_are_not_offered():
    at = new_app()
    for box in at.sidebar.selectbox:
        options = list(box.options)
        assert len(options) == len(ZONES) == 263
        assert not any("Unknown" in o or "Outside of NYC" in o for o in options)
        assert all(" - " in o for o in options)  # "Borough - Zone"
    assert "Queens - JFK Airport" in at.sidebar.selectbox[1].options


def test_zones_csv_is_clean(build_zones):
    zones = pd.read_csv(APP.parent / "zones.csv")
    assert not {264, 265} & set(zones["LocationID"])
    assert zones["label"].is_unique and zones[["lat", "lon"]].notna().all().all()
    assert list(zones.columns) == build_zones.COLUMNS


def test_the_zone_builder_drops_unknown_outside_and_centroidless_zones(build_zones):
    lookup = pd.DataFrame({"LocationID": [1, 2, 3, 4, 5],
                           "Borough": ["Queens", "Queens", "Unknown", None, "Bronx"],
                           "Zone": ["A", "A", None, "Outside of NYC", "No centroid"]})
    centroids = pd.DataFrame({"LocationID": [1, 2, 3, 4], "lat": [40.7, 40.8, 40.6, 40.5],
                              "lon": [-74.0, -73.9, -73.8, -73.7]})
    zones = build_zones.build(lookup, centroids)
    assert list(zones["LocationID"]) == [1, 2]  # 3 and 4 are Unknown/Outside, 5 has no centroid
    assert list(zones["label"]) == ["Queens - A (1)", "Queens - A (2)"]  # duplicate names get the id


@pytest.mark.parametrize("status,expected", [(400, "rejected"), (404, "rejected"), (503, "problem")])
def test_error_message_wording(ui_app, status, expected):
    assert expected in ui_app.error_message(response(status, {"detail": "x"}))


def test_haversine_known_values(ui_app):
    assert ui_app.haversine_km(40.7, -74.0, 40.7, -74.0) == 0
    assert ui_app.haversine_km(0, 0, 0, 1) == pytest.approx(111.195, abs=0.01)
