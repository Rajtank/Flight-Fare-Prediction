"""Train the flight fare RandomForest model from the notebook and save it as a pickle."""

import pickle
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = "Flight_Fare_Prediction.xlsx"
MODEL_PATH = "flight_fare_model.pkl"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.dropna()

    df["Travelled_Date"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.day
    df["Travelled_month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month
    df.drop(["Date_of_Journey"], axis=1, inplace=True)

    df["Departure_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
    df["Departure_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
    df.drop(["Dep_Time"], axis=1, inplace=True)

    df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
    df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute
    df.drop(["Arrival_Time"], axis=1, inplace=True)

    duration = list(df["Duration"])
    for i in range(len(duration)):
        if len(duration[i].split()) != 2:
            if "h" in duration[i]:
                duration[i] = duration[i].strip() + " 0m"
            else:
                duration[i] = "0h " + duration[i]

    dur_hour, dur_min = [], []
    for d in duration:
        dur_hour.append(int(re.sub("h", " ", d).split()[0]))
        dur_min.append(int(re.sub("m", " ", d).split()[-1]))

    df["duration_hour"] = dur_hour
    df["duration_min"] = dur_min
    df.drop(["Duration", "Route", "Additional_Info"], axis=1, inplace=True)

    df.replace(
        {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4},
        inplace=True,
    )
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    airline = pd.get_dummies(df[["Airline"]], drop_first=True)
    source = pd.get_dummies(df[["Source"]], drop_first=True)
    destination = pd.get_dummies(df[["Destination"]], drop_first=True)
    df = pd.concat([df, airline, source, destination], axis=1)
    df.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)
    return df


def main() -> None:
    df = encode(load_and_clean(DATA_PATH))

    x = df.drop(["Price"], axis=1)
    y = df["Price"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=12
    )

    model = RandomForestRegressor(
        n_estimators=700,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        max_depth=15,
    )
    model.fit(x_train, y_train)

    print(f"Training R^2 : {model.score(x_train, y_train):.4f}")
    print(f"Testing  R^2 : {model.score(x_test, y_test):.4f}")

    payload = {"model": model, "feature_columns": list(x.columns)}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
