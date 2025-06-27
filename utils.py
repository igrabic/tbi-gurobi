import requests
import pandas as pd
import numpy as np
from scipy import interpolate
from io import StringIO
import re


def get_consumption_profile(sample_time):
    """
    Read csv from data folder and return the consumption power profile
    """
    df = pd.read_csv(
        "./metalproduct_cons_at_EGC_2024.csv",
        usecols=[1],
        skiprows=1,
        header=None,
        names=["P"],
    )
    df["timestamp"] = pd.date_range(
        start="2024-01-01 00:00:00", end="2024-12-31 23:45:00", freq="15min"
    )
    df = df[~((df["timestamp"].dt.month == 12) & (df["timestamp"].dt.day == 31))]
    df = df.set_index("timestamp")
    df = df["P"].resample(f"{sample_time}H").mean().values.flatten()
    return df


def get_pvgis_power_array(lat, lon, azimuth, tilt, year=2020, peakpower=1.0, loss=0):
    """
    Fetches hourly PV power output and interpolates to 15-minute resolution,
    then pads with zeros to ensure exactly 35,040 samples per year.

    Returns:
        np.ndarray: Array of power values in watts (W) at 15-minute intervals (35,040 values).
    """
    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": year,
        "endyear": year,
        "pvcalculation": 1,
        "peakpower": peakpower,
        "loss": loss,
        "angle": tilt,
        "aspect": azimuth,
        "outputformat": "csv",
        "usehorizon": 1,
        "optimalangles": 0,
        "raddatabase": "PVGIS-ERA5",
        "components": 1,
        "hourlyvalues": 1,
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    content = response.text
    lines = content.splitlines()
    valid_lines = [line for line in lines if re.match(r"^\d{8}:\d{4}|^time", line)]
    cleaned_csv = "\n".join(valid_lines)

    df = pd.read_csv(StringIO(cleaned_csv), sep=",")
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", utc=True)
    if "P" not in df.columns:
        raise Exception("Power column 'P' not found in the data.")

    hourly_series = df.set_index("time")["P"]
    full_year_index = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:45:00",
        freq="15min",
        tz="UTC",
    )
    f = interpolate.interp1d(
        hourly_series.index.astype(np.int64),
        hourly_series.values,
        kind="linear",
        fill_value=0,
        bounds_error=False,
    )

    interpolated_values = f(full_year_index.astype(np.int64))
    expected_length = 35040
    if len(interpolated_values) < expected_length:
        zeros_to_add = expected_length - len(interpolated_values)
        interpolated_values = np.append(interpolated_values, np.zeros(zeros_to_add))
    elif len(interpolated_values) > expected_length:
        interpolated_values = interpolated_values[:expected_length]
    interpolated_values = np.clip(interpolated_values, 0, None)

    return interpolated_values
