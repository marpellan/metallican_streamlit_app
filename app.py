import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import os
import glob
import sqlite3
from typing import List, Tuple

# ---------- CONFIG ----------
st.set_page_config(page_title="MetalliCan Data Explorer", layout="wide")
LOGO_PATH = "logo.png"
CSV_FOLDER = "database/CSV"
SQLITE_PATH = "database/metallican.sqlite"
DEFAULT_CENTER = (56.0, -96.0)
DEFAULT_ZOOM = 4

# Maximum rows to render in the interactive dataframe to keep UI snappy
MAX_PREVIEW_ROWS = 500

# ---------- UTIL ----------
@st.cache_data
def list_csv_tables(folder: str):
    paths = glob.glob(os.path.join(folder, "*.csv"))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")

# def ensure_latlon_from_geometry(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if "geometry" in df.columns:
#         def safe_coords(val):
#             try:
#                 if pd.isna(val):
#                     return (None, None)
#                 g = wkt.loads(str(val))
#                 return (g.x, g.y)  # lon, lat
#             except Exception:
#                 return (None, None)
#         coords = df["geometry"].astype(str).apply(safe_coords)
#         df["longitude"] = coords.apply(lambda t: t[0])
#         df["latitude"] = coords.apply(lambda t: t[1])
#     return df

def ensure_latlon_from_geometry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "geometry" in df.columns:
        lats, lons = [], []
        for val in df["geometry"]:
            try:
                if pd.isna(val):
                    lats.append(None)
                    lons.append(None)
                else:
                    g = wkt.loads(str(val))
                    lons.append(float(g.x))
                    lats.append(float(g.y))
            except Exception:
                lons.append(None)
                lats.append(None)
        df["longitude"] = pd.to_numeric(lons, errors="coerce")
        df["latitude"] = pd.to_numeric(lats, errors="coerce")

        # Auto-flip safeguard: if max latitude > 90°, coordinates were swapped
        if df["latitude"].abs().max() > 90:
            df["latitude"], df["longitude"] = df["longitude"], df["latitude"]

    return df

def df_to_gdf(df: pd.DataFrame):
    if "latitude" in df.columns and "longitude" in df.columns:
        valid = df.dropna(subset=["latitude", "longitude"]).copy()
        if valid.empty:
            return None
        return gpd.GeoDataFrame(valid, geometry=gpd.points_from_xy(valid["longitude"], valid["latitude"]), crs="EPSG:4326")
    return None

def sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return c.fetchone() is not None

def sql_placeholders(n: int) -> str:
    return ",".join("?" for _ in range(n)) if n > 0 else ""

# ---------- DATA SOURCES ----------
csv_tables = list_csv_tables(CSV_FOLDER)
use_sqlite = os.path.exists(SQLITE_PATH)
sqlite_conn = None
if use_sqlite:
    sqlite_conn = sqlite3.connect(SQLITE_PATH)

# Determine dataset options (prefer table names from SQLite if available)
dataset_options = set(csv_tables.keys())
if sqlite_conn:
    # get tables from sqlite master
    cur = sqlite_conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tbls = [row[0] for row in cur.fetchall()]
    for t in tbls:
        dataset_options.add(t)
dataset_options = sorted(dataset_options)

# main table must exist
if "Main" not in dataset_options and "main_table" not in dataset_options:
    st.error("No Main table found (look for 'Main' in SQLite or 'main_table.csv' in CSV folder). Aborting.")
    st.stop()

# canonical name for main: prefer 'Main' (sqlite) else 'main_table'
MAIN_TABLE_NAME = "Main" if "Main" in dataset_options else "main_table"

# ---------- LOAD MAIN (source of filters) ----------
@st.cache_data
def load_main_table() -> pd.DataFrame:
    # from sqlite if exists and has Main
    if sqlite_conn and sqlite_table_exists(sqlite_conn, MAIN_TABLE_NAME):
        return pd.read_sql_query(f"SELECT * FROM \"{MAIN_TABLE_NAME}\"", sqlite_conn)
    # else try CSV (main_table.csv)
    key = "main_table" if "main_table" in csv_tables else None
    if key:
        return load_csv(csv_tables[key])
    raise FileNotFoundError("Main table not found in SQLite or CSVs.")

main_df = load_main_table()
main_df = ensure_latlon_from_geometry(main_df)
main_gdf = df_to_gdf(main_df)

# ---------- LAYOUT: Sidebar ----------
if os.path.exists(LOGO_PATH):
    try:
        st.sidebar.image(LOGO_PATH, width=230)
    except Exception:
        st.sidebar.image(LOGO_PATH)  # fallback

st.sidebar.title("Filters")

# Build filter options dynamically from main_df
def safe_unique_list(df, col):
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())
    return []

province_opts = safe_unique_list(main_df, "province")
commodity_opts = safe_unique_list(main_df, "commodities")
mining_opts = safe_unique_list(main_df, "mining_processing_type")
facility_type_opts = safe_unique_list(main_df, "facility_type")
status_opts = safe_unique_list(main_df, "status")

province = st.sidebar.multiselect("Province", options=province_opts)
commodity = st.sidebar.multiselect("Commodity", options=commodity_opts)
mining_type = st.sidebar.multiselect("Mining / Processing Type", options=mining_opts)
facility_type = st.sidebar.multiselect("Facility Type", options=facility_type_opts)
status = st.sidebar.multiselect("Status", options=status_opts)

# ---------- MAIN PAGE: top - search + dataset selector ----------
st.title("MetalliCan Data Explorer")
col_search, col_ds = st.columns([3, 1])
with col_search:
    search_query = st.text_input("Search by facility name, company, or commodity", placeholder="Type keywords...")
with col_ds:
    dataset_name = st.selectbox("Table to display", options=dataset_options, index=dataset_options.index(MAIN_TABLE_NAME) if MAIN_TABLE_NAME in dataset_options else 0)

# ---------- APPLY FILTERS TO MAIN ----------
filtered_main = main_df.copy()

def apply_main_filters(df):
    out = df
    if province:
        out = out[out["province"].isin(province)]
    if commodity:
        out = out[out["commodities"].isin(commodity)]
    if mining_type:
        out = out[out["mining_processing_type"].isin(mining_type)]
    if facility_type:
        out = out[out["facility_type"].isin(facility_type)]
    if status:
        out = out[out["status"].isin(status)]
    if search_query:
        q = search_query.lower()
        def row_match(r):
            return (q in str(r.get("facility_name","")).lower()
                    or q in str(r.get("company_name","")).lower()
                    or q in str(r.get("commodities","")).lower())
        out = out[out.apply(row_match, axis=1)]
    return out

filtered_main = apply_main_filters(filtered_main)
main_ids = filtered_main["main_id"].dropna().unique().tolist()

st.markdown(f"**Showing {len(filtered_main):,} facilities based on filters**")

# ---------- LOAD SELECTED DATASET (efficiently) ----------
def load_dataset_filtered_by_main(dataset: str, main_ids: List[str]) -> pd.DataFrame:
    """Load dataset and filter by main_ids efficiently using SQLite if available, else CSV filtering."""
    # normalize dataset name: SQLite may have "Main" etc.
    # If dataset is main table, just return filtered_main (all columns)
    if dataset == MAIN_TABLE_NAME or dataset.lower() in ("main", "main_table"):
        return filtered_main.copy()

    # Try sqlite
    if sqlite_conn and sqlite_table_exists(sqlite_conn, dataset):
        # If no main_ids (empty selection), return empty DataFrame
        if not main_ids:
            # return empty with table columns
            df_empty = pd.read_sql_query(f"SELECT * FROM \"{dataset}\" LIMIT 0", sqlite_conn)
            return df_empty
        # build parameterized IN clause
        placeholders = sql_placeholders(len(main_ids))
        sql = f"SELECT * FROM \"{dataset}\" WHERE main_id IN ({placeholders})"
        params = tuple(main_ids)
        df = pd.read_sql_query(sql, sqlite_conn, params=params)
        return df
    # Else fallback to CSV if exists
    key = dataset if dataset in csv_tables else None
    if key:
        df = load_csv(csv_tables[key])
        if "main_id" in df.columns and main_ids:
            df = df[df["main_id"].isin(main_ids)]
        else:
            # no main_id or empty main_ids: return either full table or empty
            if "main_id" in df.columns and not main_ids:
                return df.iloc[0:0]  # empty
        return df
    # Not found
    return pd.DataFrame()

# load dataset
dataset_df = load_dataset_filtered_by_main(dataset_name, main_ids)
dataset_df = ensure_latlon_from_geometry(dataset_df)

# If dataset is non-main but we want facility context, attempt to join extra main fields (facility_name, lat/lon)
if dataset_name != MAIN_TABLE_NAME and "main_id" in dataset_df.columns:
    # left merge with filtered_main's selected columns to get facility_name and coords
    ctx_cols = ["main_id", "facility_name", "company_name", "province", "commodities", "geometry"]
    ctx_present = [c for c in ctx_cols if c in filtered_main.columns]
    if ctx_present:
        # merge on main_id via pandas; if dataset_df large, this is limited because dataset_df was already filtered by main_ids
        dataset_df = dataset_df.merge(filtered_main[ctx_present], on="main_id", how="left", suffixes=("", "_main"))

# ---------- BUILD MAP: one marker per facility (from filtered_main) ----------
st.subheader("Map")

# Build counts per facility if viewing another dataset (to show how many records per facility)
counts_by_main = {}
if dataset_name != MAIN_TABLE_NAME and "main_id" in dataset_df.columns:
    try:
        counts_series = dataset_df["main_id"].value_counts()
        counts_by_main = counts_series.to_dict()
    except Exception:
        counts_by_main = {}

# Use filtered_main for markers (one per facility)
#map_df = filtered_main.copy()
# --- FILTER MAP FACILITIES BY SELECTED DATASET ---
if dataset_name != MAIN_TABLE_NAME and "main_id" in dataset_df.columns:
    dataset_main_ids = dataset_df["main_id"].dropna().unique().tolist()
    map_df = filtered_main[filtered_main["main_id"].isin(dataset_main_ids)].copy()
else:
    map_df = filtered_main.copy()

map_gdf = df_to_gdf(map_df)

if map_gdf is not None and not map_gdf.empty:
    center_lat = float(map_gdf["latitude"].median())
    center_lon = float(map_gdf["longitude"].median())
else:
    center_lat, center_lon = DEFAULT_CENTER

m = folium.Map(location=[center_lat, center_lon], zoom_start=DEFAULT_ZOOM, tiles="CartoDB positron")
cluster = MarkerCluster().add_to(m)

for _, row in map_df.iterrows():
    lat = row.get("latitude")
    lon = row.get("longitude")
    if pd.isna(lat) or pd.isna(lon):
        continue
    main_id = row.get("main_id", "")
    # popup: include count if dataset != main
    popup_lines = []
    popup_lines.append(f"<b>{row.get('facility_name','')}</b>")
    if "company_name" in row and pd.notna(row.get("company_name")):
        popup_lines.append(f"Company: {row.get('company_name')}")
    if "province" in row and pd.notna(row.get("province")):
        popup_lines.append(f"Province: {row.get('province')}")
    if "commodities" in row and pd.notna(row.get("commodities")):
        popup_lines.append(f"Commodity: {row.get('commodities')}")
    if dataset_name != MAIN_TABLE_NAME:
        cnt = counts_by_main.get(main_id, 0)
        popup_lines.append(f"<i>{cnt} record(s) in {dataset_name}</i>")
    popup_lines.append(f"<small style='color:gray'>main_id: {main_id}</small>")
    popup_html = "<br/>".join(popup_lines)
    folium.Marker([lat, lon], popup=folium.Popup(popup_html, max_width=350), tooltip=row.get("facility_name","")).add_to(cluster)

st_map = st_folium(m, width=None, height=650)

# ---------- TABLE BELOW MAP: show ALL columns (with preview for large datasets) ----------
st.subheader(f"Filtered table: {dataset_name} (showing up to {MAX_PREVIEW_ROWS} rows)")

if dataset_df.empty:
    st.warning("No rows match the current filters for this dataset.")
else:
    total_rows = len(dataset_df)
    if total_rows > MAX_PREVIEW_ROWS:
        st.info(f"Dataset has {total_rows:,} rows after filtering. Showing a preview of the first {MAX_PREVIEW_ROWS:,} rows to keep UI responsive.")
        preview = dataset_df.head(MAX_PREVIEW_ROWS)
        st.dataframe(preview.reset_index(drop=True), use_container_width=True)
        if st.button("Show ALL rows in table (may be slow)"):
            st.dataframe(dataset_df.reset_index(drop=True), use_container_width=True)
    else:
        st.dataframe(dataset_df.reset_index(drop=True), use_container_width=True)

    # Download filtered dataset (full)
    csv_bytes = dataset_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download full filtered table (CSV)", data=csv_bytes, file_name=f"{dataset_name}_filtered.csv", mime="text/csv")

# ---------- OPTIONAL: Show related rows for a clicked facility or selection ----------
st.markdown("---")
st.subheader("Facility details (select from map or list)")

# Try to detect map click main_id from st_map
clicked_main_id = None
if isinstance(st_map, dict):
    clicked = st_map.get("last_clicked") or st_map.get("last_object_clicked") or st_map.get("clicked")
    if clicked and "lat" in clicked and "lng" in clicked:
        # find nearest facility in filtered_main
        if "latitude" in filtered_main.columns and "longitude" in filtered_main.columns:
            clat, clng = clicked["lat"], clicked["lng"]
            d2 = (filtered_main["latitude"] - clat)**2 + (filtered_main["longitude"] - clng)**2
            if not d2.empty and d2.notna().any():
                nearest_idx = d2.idxmin()
                clicked_main_id = filtered_main.loc[nearest_idx, "main_id"]
            else:
                clicked_main_id = None

# Provide a selectbox to pick facility by name as well
facility_opts = ["-- none --"] + filtered_main["facility_name"].dropna().unique().tolist()
facility_choice = st.selectbox("Or select a facility by name", options=facility_opts, index=0)
selected_main_id = None
if clicked_main_id:
    selected_main_id = clicked_main_id
elif facility_choice != "-- none --":
    sel = filtered_main[filtered_main["facility_name"] == facility_choice]
    if not sel.empty:
        selected_main_id = sel.iloc[0]["main_id"]

if selected_main_id:
    st.markdown(f"### Selected facility: `{selected_main_id}`")
    # show main info
    main_row = filtered_main[filtered_main["main_id"] == selected_main_id]
    if not main_row.empty:
        st.write(main_row.to_dict(orient="records")[0])
    # show related rows from other datasets (small summary)
    related_tables = []
    # find up to 6 other datasets that have main_id
    for ds in dataset_options:
        if ds == MAIN_TABLE_NAME:
            continue
        # prefer sqlite check, else CSV header check
        has = False
        if sqlite_conn and sqlite_table_exists(sqlite_conn, ds):
            # check if main_id exists in schema by reading zero rows
            try:
                df0 = pd.read_sql_query(f"SELECT * FROM \"{ds}\" LIMIT 0", sqlite_conn)
                has = "main_id" in df0.columns
            except Exception:
                has = False
        elif ds in csv_tables:
            try:
                df0 = load_csv(csv_tables[ds], )
                has = "main_id" in df0.columns
            except Exception:
                has = False
        if has:
            related_tables.append(ds)
    st.markdown("**Related tables (counts)**")
    for rt in related_tables:
        count_val = 0
        try:
            if sqlite_conn and sqlite_table_exists(sqlite_conn, rt):
                cur = sqlite_conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM \"{rt}\" WHERE main_id=?;", (selected_main_id,))
                count_val = cur.fetchone()[0]
            elif rt in csv_tables:
                df_rt = load_csv(csv_tables[rt])
                if "main_id" in df_rt.columns:
                    count_val = int((df_rt["main_id"] == selected_main_id).sum())
        except Exception:
            count_val = 0
        st.write(f"- {rt}: {count_val:,} row(s)")

else:
    st.info("Select a facility on the map or from the dropdown to see details and related tables.")

# ---------- footer ----------
st.markdown("---")
st.caption("Marin Pellan© 2025 MetalliCan Data Explorer")
