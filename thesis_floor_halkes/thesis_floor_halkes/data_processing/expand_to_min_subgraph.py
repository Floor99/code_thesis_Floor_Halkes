import pandas as pd
import numpy as np


def expand_wait_times(df, num_peaks=2, amp_frac=0.1, sigma=1.0):
    """
    Expand a DataFrame of 15-minute averaged wait_times into per-minute resolution
    with small Gaussian bumps for realistic variation.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame with columns ['node_id', 'timestamp', 'wait_time'] where
        'timestamp' is a Timestamp at 15-minute intervals.
    num_peaks : int, default=2
        Number of random Gaussian bumps to add per 15-minute block.
    amp_frac : float, default=0.1
        Maximum bump height as a fraction of the block's start wait_time.
    sigma : float, default=1.0
        Width (in minutes) of each Gaussian bump.

    Returns:
    -------
    pd.DataFrame
        Expanded DataFrame at 1-minute resolution with columns
        ['node_id', 'timestamp', 'wait_time'].
    """
    # Make a working copy and extract date
    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    df2["date"] = df2["timestamp"].dt.date

    def expand_block(avg_k, avg_k1):
        t = np.arange(15)
        # 1) linear baseline
        baseline = avg_k + (avg_k1 - avg_k) * (t / 15.0)

        # 2) add Gaussian bumps
        peaks = np.zeros_like(t, dtype=float)
        for _ in range(num_peaks):
            center = np.random.uniform(0, 15)
            try:
                height = np.random.uniform(0, amp_frac * avg_k)
            except Exception as e:
                print(f"{amp_frac=}, {avg_k=}, {avg_k1=}, {center=}")
                raise e
            peaks += height * np.exp(-0.5 * ((t - center) / sigma) ** 2)
                    
        peaks[0] = 0.0  # ensure the first minute stays exact

        # 3) combine
        y_raw = baseline + peaks

        # 4) correct only the in-between minutes so their mean matches avg_k
        delta = y_raw[1:].mean() - avg_k
        y = y_raw.copy()
        y[1:] = y_raw[1:] - delta

        # 5) clip negatives
        return np.clip(y, 0, None)

    rows = []

    # Group by node and day to avoid crossing midnight
    for (node, day), group in df2.groupby(["node_id", "date"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for i, row in group.iterrows():
            avg_k = row["wait_time"]
            if not np.isfinite(avg_k):
                avg_k = 20
            avg_k1 = group.loc[i + 1, "wait_time"] if i < len(group) - 1 else avg_k
            start_ts = row["timestamp"]
            y = expand_block(avg_k, avg_k1)

            # build 15 per-minute rows
            for offset, wt in enumerate(y):
                row_data = row.drop("date").to_dict()
                row_data["timestamp"] = start_ts + pd.Timedelta(minutes=offset)
                row_data["wait_time"] = wt
                rows.append(row_data)

    # Assemble final DataFrame
    new_df = pd.DataFrame(rows)
    new_df = new_df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    return new_df


if __name__ == "__main__":
    # time_series = pd.read_parquet("data/processed/node_features.parquet")
    timeseries_subgraph = pd.read_parquet(
        "data/processed_new/timeseries_subgraph.parquet"
    )
   
    new_df = expand_wait_times(
        timeseries_subgraph, num_peaks=3, amp_frac=0.1, sigma=1.5
    )
    new_df.to_parquet("data/processed_new/timeseries_expanded_subgraph.parquet")
