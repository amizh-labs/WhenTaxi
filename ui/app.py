"""Streamlit demo UI for the ETA API. Demo only: it is not part of the platform and nothing in src/ imports it.

Run:  streamlit run ui/app.py        (API_URL env var, default http://localhost:8000)
The UI only ever sends {pickup_zone, dropoff_zone, pickup_ts} to POST /predict. The straight-line distance shown
next to the prediction is computed here from ui/zones.csv and is never sent to the API.
"""
import datetime as dt
import math
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ZONES_CSV = Path(__file__).resolve().parent / "zones.csv"
TIMEOUT_SECONDS = 10
EARTH_RADIUS_KM = 6371.0088
DEFAULT_PICKUP, DEFAULT_DROPOFF = 161, 132  # Midtown Center, JFK Airport
DEFAULT_DATE, DEFAULT_TIME = dt.date(2026, 3, 15), dt.time(18, 0)
REPLAY_NOTE = ("Demo on historical data: the model's rolling traffic features come from a stored snapshot, so this "
               "is a replay demo, not a live ETA service.")
HISTORY_COLUMNS = ["pickup", "dropoff", "pickup time", "eta (min)", "model", "weather", "distance (km)"]


def api_url() -> str:
    return os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


@st.cache_data
def load_zones() -> pd.DataFrame:
    return pd.read_csv(ZONES_CSV).set_index("LocationID")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def error_message(response: requests.Response) -> str:
    """A plain-language message for a non-200 response."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors: [{"loc": [...], "msg": "..."}]
        detail = "; ".join(f"{'.'.join(str(p) for p in d.get('loc', []))}: {d.get('msg', '')}" for d in detail)
    suffix = f": {detail}" if detail else ""
    if 400 <= response.status_code < 500:
        return f"The API rejected this request (HTTP {response.status_code}){suffix}. Check the zones and the time."
    return f"The API had a problem (HTTP {response.status_code}){suffix}. Try again in a moment."


def predict(pickup_zone: int, dropoff_zone: int, pickup_ts: dt.datetime) -> tuple[dict | None, str | None]:
    """Returns (response json, None) or (None, error message). Never raises for network or HTTP problems."""
    payload = {"pickup_zone": pickup_zone, "dropoff_zone": dropoff_zone, "pickup_ts": pickup_ts.isoformat()}
    try:
        response = requests.post(f"{api_url()}/predict", json=payload, timeout=TIMEOUT_SECONDS)
    except requests.Timeout:
        return None, f"The API did not answer within {TIMEOUT_SECONDS} seconds. Try again."
    except requests.ConnectionError:
        return None, f"Could not reach the API at {api_url()}. Is it running?"
    except requests.RequestException as exc:
        return None, f"The request failed: {exc}"
    if response.status_code != 200:
        return None, error_message(response)
    try:
        return response.json(), None
    except ValueError:
        return None, "The API answered with something that is not JSON."


def main() -> None:
    st.set_page_config(page_title="Taxi ETA demo", layout="wide")
    zones = load_zones()
    ids = list(zones.index)
    labels = zones["label"]

    def index_of(zone_id: int) -> int:
        return ids.index(zone_id) if zone_id in ids else 0

    st.title("NYC taxi trip ETA")
    st.info(REPLAY_NOTE)

    with st.sidebar:
        st.header("Trip")
        pickup = st.selectbox("Pickup zone", ids, index=index_of(DEFAULT_PICKUP), format_func=labels.get)
        dropoff = st.selectbox("Dropoff zone", ids, index=index_of(DEFAULT_DROPOFF), format_func=labels.get)
        date = st.date_input("Pickup date", value=DEFAULT_DATE)
        time = st.time_input("Pickup time", value=DEFAULT_TIME)
        clicked = st.button("Predict ETA", type="primary")
        st.caption(f"API: {api_url()}")

    history = st.session_state.setdefault("history", [])
    if clicked:
        pickup_ts = dt.datetime.combine(date, time)
        result, error = predict(int(pickup), int(dropoff), pickup_ts)
        if error:
            st.session_state["last"] = None
            st.error(error)
        else:
            a, b = zones.loc[pickup], zones.loc[dropoff]
            km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            st.session_state["last"] = {"result": result, "km": km, "pickup": int(pickup), "dropoff": int(dropoff)}
            history.insert(0, {
                "pickup": a["label"], "dropoff": b["label"], "pickup time": pickup_ts.strftime("%Y-%m-%d %H:%M"),
                "eta (min)": round(result["eta_minutes"], 1), "model": result["model_version"],
                "weather": result["weather_source"], "distance (km)": round(km, 2)})

    last = st.session_state.get("last")
    if last:
        result = last["result"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ETA (minutes)", f"{result['eta_minutes']:.1f}")
        c2.metric("Model version", str(result["model_version"]))
        c3.metric("Weather source", str(result["weather_source"]))
        c4.metric("Distance (km)", f"{last['km']:.2f}", help="Straight line between the zone centroids, computed in the UI")
        points = zones.loc[[last["pickup"], last["dropoff"]], ["lat", "lon"]]
        st.map(points, size=200)
    elif not clicked:
        st.write("Pick a pickup zone, a dropoff zone and a time in the sidebar, then press **Predict ETA**.")

    if history:
        st.subheader("This session")
        st.dataframe(pd.DataFrame(history, columns=HISTORY_COLUMNS), hide_index=True, use_container_width=True)


if __name__ == "__main__":  # streamlit and AppTest both run the script as __main__; tests can import it
    main()
