import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("data/raw/VialisStatistics_2025-04-14_2025-04-20_INTERSECTION.csv", sep=";")
    df.columns = df.columns.str.strip().str.lower()
    df["version"] = df["is_name"].str[-1].astype(int)

    idx = df.groupby("tlc_name")["version"].transform(max) == df["version"]
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    df[idx].to_csv(
        "data/processed/intersection_measurements_week.csv", sep=",", index=False
    )

