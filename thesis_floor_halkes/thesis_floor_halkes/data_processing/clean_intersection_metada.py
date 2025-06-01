import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("data/raw/intersection_metadata.csv", sep=";")
    df.columns = df.columns.str.strip().str.lower()
    df[["lat", "lon"]] = df["coördinaten"].str.split(" , ").to_list()
    df = df.drop(columns=["coördinaten"])
    df = df.rename(columns={"naam": "tlc_name"})
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.to_csv("data/processed/intersection_metadata.csv", sep=",", index=False)
