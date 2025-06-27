from os import name
import pandas as pd


def get_day_ahead_check_data():
    """
    Reads the day-ahead check data from a CSV file and returns it as a DataFrame.
    """
    df = pd.read_csv(
        "./day-ahead_prices_CRO_2024.csv",
        skiprows=1,
        sep=";",
        names=["timestamp", "price"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y %H:%M")
    df.set_index("timestamp", inplace=True)
    # df = df.resample("{}h".format(1)).mean()
    df["price"] = df.price / 1000  # Convert price from €/MWh to €/kWh
    prices = df.price.values.flatten()[: 1 * 24 * 365]  # Return only the first year
    return df


if __name__ == "__main__":
    # Example usage
    print("Fetching day-ahead check data...")
    day_ahead_data = get_day_ahead_check_data()
    print("collumns", day_ahead_data.columns)
    print(
        day_ahead_data["price"].isna().sum(),
        "NaN values found in the data.",
    )
    print(
        (day_ahead_data["price"] == float("inf")).sum(),
        "inf values found in the data.",
    )
    print(
        (day_ahead_data["price"] == float("-inf")).sum(),
        "-inf values found in the data.",
    )
    # You can add more processing or analysis here as needed
    print(day_ahead_data["price"].values[:-24].shape, 24 * 365)
    print("Indices with NaN values:")
    print(day_ahead_data.index[day_ahead_data["price"].isna()])
