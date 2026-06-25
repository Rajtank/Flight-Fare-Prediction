"""Streamlit app for the Flight Fare Prediction model trained in the notebook."""

import os
import pickle
from datetime import date, time

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "flight_fare_model.pkl"

AIRLINES = [
    "Air Asia",
    "Air India",
    "GoAir",
    "IndiGo",
    "Jet Airways",
    "Jet Airways Business",
    "Multiple carriers",
    "Multiple carriers Premium economy",
    "SpiceJet",
    "Trujet",
    "Vistara",
    "Vistara Premium economy",
]

SOURCES = ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai"]

DESTINATIONS = ["Banglore", "Cochin", "Delhi", "Hyderabad", "Kolkata", "New Delhi"]

STOPS_MAP = {
    "Non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4,
}


@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_feature_row(
    feature_columns: list[str],
    *,
    total_stops: int,
    travelled_date: int,
    travelled_month: int,
    dep_hour: int,
    dep_min: int,
    arr_hour: int,
    arr_min: int,
    duration_hour: int,
    duration_min: int,
    airline: str,
    source: str,
    destination: str,
) -> pd.DataFrame:
    row = {col: 0 for col in feature_columns}
    row["Total_Stops"] = total_stops
    row["Travelled_Date"] = travelled_date
    row["Travelled_month"] = travelled_month
    row["Departure_hour"] = dep_hour
    row["Departure_min"] = dep_min
    row["Arrival_hour"] = arr_hour
    row["Arrival_min"] = arr_min
    row["duration_hour"] = duration_hour
    row["duration_min"] = duration_min

    airline_col = f"Airline_{airline}"
    if airline_col in row:
        row[airline_col] = 1

    source_col = f"Source_{source}"
    if source_col in row:
        row[source_col] = 1

    dest_col = f"Destination_{destination}"
    if dest_col in row:
        row[dest_col] = 1

    return pd.DataFrame([row], columns=feature_columns)


def compute_duration(dep_h: int, dep_m: int, arr_h: int, arr_m: int) -> tuple[int, int]:
    dep_total = dep_h * 60 + dep_m
    arr_total = arr_h * 60 + arr_m
    diff = arr_total - dep_total
    if diff < 0:
        diff += 24 * 60
    return diff // 60, diff % 60


st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="centered")

st.title("✈️ Flight Fare Predictor")
st.write(
    "Predict the price of a flight ticket using the Random Forest model "
    "trained in `Flight_Fare_Analysis.ipynb`."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file `{MODEL_PATH}` not found. "
        "Run `python train_model.py` first to generate it."
    )
    st.stop()

payload = load_model(MODEL_PATH)
model = payload["model"]
feature_columns = payload["feature_columns"]

with st.form("fare_form"):
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox("Airline", AIRLINES, index=AIRLINES.index("IndiGo"))
        source = st.selectbox("Source", SOURCES, index=SOURCES.index("Delhi"))
        destination = st.selectbox(
            "Destination", DESTINATIONS, index=DESTINATIONS.index("Cochin")
        )
        stops_label = st.selectbox("Total Stops", list(STOPS_MAP.keys()), index=1)

    with col2:
        journey_date = st.date_input("Date of Journey", value=date(2019, 6, 1))
        dep_time = st.time_input("Departure Time", value=time(10, 0))
        arr_time = st.time_input("Arrival Time", value=time(13, 0))

    submitted = st.form_submit_button("Predict Fare")

if submitted:
    if source == destination:
        st.warning("Source and Destination cannot be the same.")
        st.stop()

    dur_h, dur_m = compute_duration(
        dep_time.hour, dep_time.minute, arr_time.hour, arr_time.minute
    )

    X = build_feature_row(
        feature_columns,
        total_stops=STOPS_MAP[stops_label],
        travelled_date=journey_date.day,
        travelled_month=journey_date.month,
        dep_hour=dep_time.hour,
        dep_min=dep_time.minute,
        arr_hour=arr_time.hour,
        arr_min=arr_time.minute,
        duration_hour=dur_h,
        duration_min=dur_m,
        airline=airline,
        source=source,
        destination=destination,
    )

    price = float(model.predict(X)[0])

    st.success(f"Estimated Fare: ₹ {price:,.0f}")
    st.caption(f"Flight duration used: {dur_h}h {dur_m}m")

    with st.expander("Feature vector sent to model"):
        st.dataframe(X.T.rename(columns={0: "value"}))
