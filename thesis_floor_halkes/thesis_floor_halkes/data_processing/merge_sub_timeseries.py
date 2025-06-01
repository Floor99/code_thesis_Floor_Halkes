import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import BallTree


def load_metadata(meta_path: str) -> pd.DataFrame:
    """Load intersection metadata from CSV."""
    return pd.read_csv(meta_path)


def load_nodes(nodes_path: str) -> pd.DataFrame:
    """Load node coordinates from Parquet."""
    return pd.read_parquet(nodes_path)


def load_measurements(meas_path: str) -> pd.DataFrame:
    """Load and pre-filter intersection measurements CSV, dropping TEL entries."""
    df = pd.read_csv(meas_path, parse_dates=['timestamp'])
    return df[~df['tlc_name'].str.contains('TEL', na=False)]


def merge_light_data(df_meta: pd.DataFrame, df_meas: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata with measurements to get light locations and their wait times."""
    df = (
        df_meta
        .merge(
            df_meas[['tlc_name', 'wait_time_all_cycles_average']],
            on='tlc_name', how='inner'
        )
        .rename(columns={'wait_time_all_cycles_average': 'wait_time'})
    )
    return df[['tlc_name', 'lat', 'lon', 'wait_time']]


def build_geodataframes(
    df_lights: pd.DataFrame,
    df_nodes: pd.DataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Construct GeoDataFrames for lights and nodes in WGS84."""
    gdf_lights = gpd.GeoDataFrame(
        df_lights,
        geometry=[Point(xy) for xy in zip(df_lights.lon, df_lights.lat)],
        crs='EPSG:4326'
    )
    gdf_nodes = gpd.GeoDataFrame(
        df_nodes,
        geometry=[Point(xy) for xy in zip(df_nodes.lon, df_nodes.lat)],
        crs='EPSG:4326'
    )
    return gdf_lights, gdf_nodes


def find_nearest_lights(
    gdf_lights: gpd.GeoDataFrame,
    gdf_nodes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Assign to each node its nearest light and compute distance in meters."""
    coords_lights = np.radians(gdf_lights[['lat', 'lon']].values)
    coords_nodes = np.radians(gdf_nodes[['lat', 'lon']].values)
    tree = BallTree(coords_lights, metric='haversine')

    dist_rad, idx = tree.query(coords_nodes, k=1)
    dist_m = dist_rad.flatten() * 6_371_000

    gdf = gdf_nodes.copy()
    gdf['nearest_light_idx'] = idx.flatten()
    gdf['distance_m'] = dist_m
    gdf['tlc_name'] = gdf_lights.iloc[gdf['nearest_light_idx']].tlc_name.values
    gdf['wait_time'] = gdf_lights.iloc[gdf['nearest_light_idx']].wait_time.values
    return gdf


def apply_threshold(
    gdf_nodes: gpd.GeoDataFrame,
    threshold: float = 25
) -> gpd.GeoDataFrame:
    """Zero-out lights farther than threshold and flag nodes with a nearby light."""
    gdf = gdf_nodes.copy()
    mask = gdf['distance_m'] <= threshold
    gdf.loc[~mask, ['tlc_name', 'wait_time']] = 0
    gdf['has_light'] = mask.astype(int)
    return gdf


def build_nodes_map(
    gdf_nodes: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Extract static node-to-light lookup information."""
    return gdf_nodes[['node_id', 'osmid_original', 'tlc_name', 'lat', 'lon', 'has_light', 'distance_m']].copy()


def build_time_grid(
    df_meas: pd.DataFrame
) -> pd.DataFrame:
    """Create a full-day 15-minute interval timestamp grid."""
    day = df_meas['timestamp'].dt.normalize().min()
    times = pd.date_range(
        start=day,
        end=day + pd.Timedelta(hours=23, minutes=45),
        freq='15T'
    )
    return pd.DataFrame({'timestamp': times})


def cross_join(
    df_nodes_map: pd.DataFrame,
    df_times: pd.DataFrame
) -> pd.DataFrame:
    """Perform a Cartesian product of nodes and time grid."""
    nm = df_nodes_map.copy()
    tm = df_times.copy()
    nm['key'] = 1
    tm['key'] = 1
    df_cross = nm.merge(tm, on='key').drop(columns='key')
    return df_cross


def build_final_df(
    df_cross: pd.DataFrame,
    df_meas: pd.DataFrame,
    min_wait_time_no_light: float = 5.0,
    max_wait_time_no_light: float = 15.0
) -> pd.DataFrame:
    """Merge with real measurements and fill missing wait times with peer averages."""
    np.random.seed(42) 
    df = (
        df_cross
        .merge(
            df_meas[['tlc_name', 'timestamp', 'wait_time_all_cycles_average']],
            on=['tlc_name', 'timestamp'], how='left'
        )
        .rename(columns={'wait_time_all_cycles_average': 'wait_time'})
        .assign(
            peer_avg=lambda d: d.groupby('timestamp')['wait_time'].transform('mean'),
            wait_time=lambda d: d['wait_time'].fillna(d['peer_avg'])
        )
        .drop(columns='peer_avg')
        .sort_values(['node_id', 'timestamp'])
        .reset_index(drop=True)
    )
    df['tlc_name'] = df['tlc_name'].astype('string')
    # df.loc[df['has_light'] == 0, 'wait_time'] = 0.0
    df.loc[df['has_light'] == 0, 'wait_time'] = np.random.uniform(min_wait_time_no_light, max_wait_time_no_light, size=(df['has_light'] == 0).sum())
    
    if df['wait_time'].isnull().any():
        print("Warning: Some wait times are still NaN after merging. This may indicate missing data for some lights at certain timestamps.")
        df['wait_time']= df['wait_time'].fillna(df['wait_time'].mean())  
    
    return df


def merge_timeseries_pipeline(
    meta_path: str,
    nodes_path: str,
    meas_path: str,
    threshold: float = 25
) -> pd.DataFrame:
    """
    Full pipeline: loads data, computes nearest lights, builds time-expanded features, and writes output.
    Returns the final DataFrame.
    """
    df_meta = load_metadata(meta_path)
    df_nodes = load_nodes(nodes_path)
    df_meas = load_measurements(meas_path)

    df_lights = merge_light_data(df_meta, df_meas)
    gdf_lights, gdf_nodes = build_geodataframes(df_lights, df_nodes)
    gdf_nodes = find_nearest_lights(gdf_lights, gdf_nodes)
    gdf_nodes = apply_threshold(gdf_nodes, threshold)

    df_nodes_map = build_nodes_map(gdf_nodes)
    df_times = build_time_grid(df_meas)
    df_cross = cross_join(df_nodes_map, df_times)
    df_final = build_final_df(df_cross, df_meas)

    
    return df_final


if __name__ == "__main__":
    meta_path = "data/processed/intersection_metadata.csv"
    nodes_path = "data/processed_new/subgraph_nodes.parquet"
    meas_path = "data/processed/intersection_measurements_31_01_24.csv"
    threshold = 25 
    
    df_final = merge_timeseries_pipeline(
        meta_path,
        nodes_path,
        meas_path,
        threshold
    )
    
    print(df_final)
    
    