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
MAX_PREVIEW_ROWS = 500
MAX_MAP_POINTS = 500  # safeguard for number of markers on the map

# ---------- UTIL ----------
@st.cache_data
def list_csv_tables(folder: str):
    paths = glob.glob(os.path.join(folder, "*.csv"))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")

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
        # safety: if latitude values look like longitudes (abs>90), swap
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

# # ---------- DATA SOURCES ----------
# csv_tables = list_csv_tables(CSV_FOLDER)
# use_sqlite = os.path.exists(SQLITE_PATH)
# sqlite_conn = sqlite3.connect(SQLITE_PATH) if use_sqlite else None
#
# dataset_options = set(csv_tables.keys())
# if sqlite_conn:
#     cur = sqlite_conn.cursor()
#     cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
#     for row in cur.fetchall():
#         dataset_options.add(row[0])
# # exclude irrelevant tables
# excluded_tables = {"Sources", "Substances", "Prioritized_conservation_areas"}
# dataset_options = sorted([t for t in dataset_options if t not in excluded_tables])

# ---------- DATA SOURCES ----------
csv_tables = list_csv_tables(CSV_FOLDER)
use_sqlite = os.path.exists(SQLITE_PATH)
sqlite_conn = sqlite3.connect(SQLITE_PATH) if use_sqlite else None

dataset_options = set(csv_tables.keys())

if sqlite_conn:
    cur = sqlite_conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for row in cur.fetchall():
        dataset_options.add(row[0])

# normalize names and remove near-duplicates
def normalize_name(name: str) -> str:
    n = name.lower().replace("_table", "")
    return n

unique_datasets = {}
for name in dataset_options:
    key = normalize_name(name)
    # prefer capitalized SQLite name if both exist
    if key not in unique_datasets or name[0].isupper():
        unique_datasets[key] = name

# final sorted list, excluding irrelevant
excluded_tables = {"Sources", "Substances", "Prioritized_conservation_areas"}
dataset_options = sorted(
    [v for k, v in unique_datasets.items() if v not in excluded_tables]
)


# main table
if "Main" not in dataset_options and "main_table" not in dataset_options:
    st.error("No Main table found (look for 'Main' in SQLite or 'main_table.csv' in CSV folder). Aborting.")
    st.stop()
MAIN_TABLE_NAME = "Main" if "Main" in dataset_options else "main_table"

# ---------- LOAD MAIN ----------
@st.cache_data
def load_main_table() -> pd.DataFrame:
    if sqlite_conn and sqlite_table_exists(sqlite_conn, MAIN_TABLE_NAME):
        return pd.read_sql_query(f'SELECT * FROM "{MAIN_TABLE_NAME}"', sqlite_conn)
    key = "main_table" if "main_table" in csv_tables else None
    if key:
        return load_csv(csv_tables[key])
    raise FileNotFoundError("Main table not found in SQLite or CSVs.")

main_df = ensure_latlon_from_geometry(load_main_table())
main_gdf = df_to_gdf(main_df)

# ---------- SIDEBAR ----------
if os.path.exists(LOGO_PATH):
    try:
        st.sidebar.image(LOGO_PATH, width=230)
    except Exception:
        st.sidebar.image(LOGO_PATH)

st.sidebar.title("Filters")

def safe_unique_list(df, col):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

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

# extra overlay for land occupation only
st.sidebar.markdown("---")
show_land = st.sidebar.checkbox("Show Land Occupation polygons (Land_occupation)", value=False)

# ---------- MAIN UI ----------
st.title("MetalliCan Data Explorer")
col_search, col_ds = st.columns([3, 1])
with col_search:
    search_query = st.text_input("Search by facility name, company, or commodity", placeholder="Type keywords...")
with col_ds:
    dataset_name = st.selectbox("Table to display", options=dataset_options, index=dataset_options.index(MAIN_TABLE_NAME))

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
        out = out[
            out.apply(
                lambda r: (
                    q in str(r.get("facility_name", "")).lower()
                    or q in str(r.get("company_name", "")).lower()
                    or q in str(r.get("commodities", "")).lower()
                ),
                axis=1,
            )
        ]
    return out

filtered_main = apply_main_filters(main_df)
main_ids = filtered_main["main_id"].dropna().unique().tolist()

# ---------- LOAD SELECTED DATA ----------
def load_dataset_filtered_by_main(dataset: str, main_ids: List[str]) -> pd.DataFrame:
    if dataset == MAIN_TABLE_NAME or dataset.lower() in ("main", "main_table"):
        return filtered_main.copy()
    if sqlite_conn and sqlite_table_exists(sqlite_conn, dataset):
        if not main_ids:
            return pd.read_sql_query(f'SELECT * FROM "{dataset}" LIMIT 0', sqlite_conn)
        placeholders = sql_placeholders(len(main_ids))
        return pd.read_sql_query(f'SELECT * FROM "{dataset}" WHERE main_id IN ({placeholders})', sqlite_conn, params=tuple(main_ids))
    key = dataset if dataset in csv_tables else None
    if key:
        df = load_csv(csv_tables[key])
        if "main_id" in df.columns and main_ids:
            df = df[df["main_id"].isin(main_ids)]
        elif "main_id" in df.columns and not main_ids:
            return df.iloc[0:0]
        return df
    return pd.DataFrame()

dataset_df = ensure_latlon_from_geometry(load_dataset_filtered_by_main(dataset_name, main_ids))

if dataset_name != MAIN_TABLE_NAME and "main_id" in dataset_df.columns:
    ctx_cols = ["main_id", "facility_name", "company_name", "province", "commodities", "geometry"]
    ctx_present = [c for c in ctx_cols if c in filtered_main.columns]
    if ctx_present:
        dataset_df = dataset_df.merge(filtered_main[ctx_present], on="main_id", how="left", suffixes=("", "_main"))

# ---------- MAP ----------
st.subheader("Map")

# Decide which facilities to show on the map depending on dataset
if dataset_name != MAIN_TABLE_NAME and "main_id" in dataset_df.columns:
    dataset_main_ids = dataset_df["main_id"].dropna().unique().tolist()
    map_df = filtered_main[filtered_main["main_id"].isin(dataset_main_ids)].copy()
else:
    map_df = filtered_main.copy()

# Update message based on actual map_df
st.markdown(f"**Showing {len(map_df):,} facilities displayed on map**")

# Performance: limit markers
if len(map_df) > MAX_MAP_POINTS:
    st.info(f"Displaying first {MAX_MAP_POINTS} facilities on map for performance.")
    map_df = map_df.head(MAX_MAP_POINTS)

map_gdf = df_to_gdf(map_df)
if map_gdf is not None and not map_gdf.empty:
    center_lat = float(map_gdf["latitude"].median())
    center_lon = float(map_gdf["longitude"].median())
else:
    center_lat, center_lon = DEFAULT_CENTER

m = folium.Map(location=[center_lat, center_lon], zoom_start=DEFAULT_ZOOM, tiles="CartoDB positron")
cluster = MarkerCluster().add_to(m)

# Use CircleMarker to avoid icon rendering issues
# for _, row in map_df.iterrows():
#     lat, lon = row.get("latitude"), row.get("longitude")
#     if pd.isna(lat) or pd.isna(lon):
#         continue
#     main_id = row.get("main_id", "")
#     popup_html = "<br/>".join([
#         f"<b>{row.get('facility_name','')}</b>",
#         f"Company: {row.get('company_name','')}" if pd.notna(row.get('company_name')) else "",
#         f"Province: {row.get('province','')}" if pd.notna(row.get('province')) else "",
#         f"Commodity: {row.get('commodities','')}" if pd.notna(row.get('commodities')) else "",
#         f"<small style='color:gray'>main_id: {main_id}</small>"
#     ])
#     folium.CircleMarker(
#         location=[lat, lon],
#         radius=5,
#         fill=True,
#         fill_opacity=0.9,
#         color="#1f78b4",
#         fill_color="#1f78b4",
#         popup=folium.Popup(popup_html, max_width=350),
#         tooltip=row.get("facility_name", "")
#     ).add_to(cluster)

# ---------- SYMBOLS / COLORS PER FACILITY TYPE ----------
FACILITY_STYLE = {
    "mining": {"color": "#1f78b4", "emoji": "⛏️"},
    "manufacturing": {"color": "#ff7f00", "emoji": "🏭"},
    "project": {"color": "#6a3d9a", "emoji": "🚧"},
    "processing": {"color": "#33a02c", "emoji": "⚙️"},
    "exploration": {"color": "#a6cee3", "emoji": "🔍"},
    "Other": {"color": "#b15928", "emoji": "📍"}
}


# Use CircleMarker for facilities with emoji tooltip
# for _, row in map_df.iterrows():
#     lat, lon = row.get("latitude"), row.get("longitude")
#     if pd.isna(lat) or pd.isna(lon):
#         continue
#
#     ftype = str(row.get("facility_type", "Other"))
#     style = FACILITY_STYLE.get(ftype, FACILITY_STYLE["Other"])
#
#     emoji = style["emoji"]
#     color = style["color"]
#
#     main_id = row.get("main_id", "")
#     popup_html = "<br/>".join([
#         f"<b>{emoji} {row.get('facility_name','')}</b>",
#         f"Type: {ftype}",
#         f"Company: {row.get('company_name','')}" if pd.notna(row.get('company_name')) else "",
#         f"Province: {row.get('province','')}" if pd.notna(row.get('province')) else "",
#         f"Commodity: {row.get('commodities','')}" if pd.notna(row.get('commodities')) else "",
#         f"<small style='color:gray'>main_id: {main_id}</small>"
#     ])
#
#     folium.CircleMarker(
#         location=[lat, lon],
#         radius=6,
#         fill=True,
#         fill_opacity=0.9,
#         color=color,
#         fill_color=color,
#         popup=folium.Popup(popup_html, max_width=350),
#         tooltip=f"{emoji} {row.get('facility_name', '')}"
#     ).add_to(cluster)

# ---------- FACILITY MARKERS WITH EMOJI ICONS ----------
for _, row in map_df.iterrows():
    lat, lon = row.get("latitude"), row.get("longitude")
    if pd.isna(lat) or pd.isna(lon):
        continue

    ftype = str(row.get("facility_type", "Other")).strip().lower()
    style = FACILITY_STYLE.get(ftype, FACILITY_STYLE["Other"])

    emoji = style["emoji"]
    color = style["color"]
    main_id = row.get("main_id", "")

    popup_html = "<br/>".join([
        f"<b>{emoji} {row.get('facility_name','')}</b>",
        f"Type: {ftype.capitalize()}",
        f"Company: {row.get('company_name','')}" if pd.notna(row.get('company_name')) else "",
        f"Province: {row.get('province','')}" if pd.notna(row.get('province')) else "",
        f"Commodity: {row.get('commodities','')}" if pd.notna(row.get('commodities')) else "",
        f"<small style='color:gray'>main_id: {main_id}</small>"
    ])

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=350),
        tooltip=f"{emoji} {row.get('facility_name', '')}",
        icon=folium.DivIcon(
            html=f"""
            <div style="
                font-size: 18px;
                line-height: 1;
                text-align: center;
                transform: translate(-50%, -50%);
            ">{emoji}</div>
            """
        )
    ).add_to(cluster)

# ---------- LAND OCCUPATION LAYER (cached, filtered by main_id) ----------
@st.cache_resource
def load_land_geojson_for_main_ids(main_ids_tuple: Tuple[str], simplify_tol: float = 0.01):
    """
    Load Land_occupation rows whose main_id is in main_ids_tuple,
    convert geometry WKT -> shapely -> GeoDataFrame, simplify geometries,
    and return a GeoJSON string and a DataFrame with properties.
    """
    if not sqlite_conn or not sqlite_table_exists(sqlite_conn, "Land_occupation"):
        return None  # table not present
    if not main_ids_tuple:
        return None
    try:
        placeholders = sql_placeholders(len(main_ids_tuple))
        sql = f'SELECT land_occupation_id, main_id, geometry FROM "Land_occupation" WHERE main_id IN ({placeholders}) AND geometry IS NOT NULL'
        df = pd.read_sql_query(sql, sqlite_conn, params=tuple(main_ids_tuple))
        if df.empty:
            return None
        # convert
        df["geometry"] = df["geometry"].astype(str)
        gdf = gpd.GeoDataFrame(df, geometry=df["geometry"].apply(wkt.loads), crs="EPSG:4326")
        # simplify to reduce payload
        gdf["geometry"] = gdf["geometry"].simplify(simplify_tol, preserve_topology=True)
        # return the GeoDataFrame (we'll call .to_json() when adding to the map)
        return gdf
    except Exception as e:
        # don't crash the app; return None so UI can show warning
        st.warning(f"Error loading Land_occupation geometries: {e}")
        return None

# Only load and add land polygons if user ticked the box AND we have main_ids to restrict to
if show_land:
    # build tuple of main_ids to ask the loader for
    if dataset_name != MAIN_TABLE_NAME and "main_id" in dataset_df.columns:
        # use only the main_ids present in the selected dataset (not entire main list)
        main_ids_for_land = tuple(sorted(set(dataset_df["main_id"].dropna().astype(str).tolist())))
    else:
        # all filtered_main main_ids
        main_ids_for_land = tuple(sorted(set(filtered_main["main_id"].dropna().astype(str).tolist())))
    # If empty, skip
    if main_ids_for_land:
        land_gdf = load_land_geojson_for_main_ids(main_ids_for_land, simplify_tol=0.01)
        if land_gdf is None or land_gdf.empty:
            st.info("No Land_occupation polygons to display for the selected facilities.")
        else:
            # Prepare popup/tooltip fields: show land_occupation_id and main_id
            try:
                folium.GeoJson(
                    data=land_gdf.to_json(),
                    name="Land_occupation",
                    style_function=lambda feat: {
                        "fillColor": "#33a02c",
                        "color": "#33a02c",
                        "fillOpacity": 0.35,
                        "weight": 1,
                    },
                    tooltip=folium.GeoJsonTooltip(fields=["land_occupation_id", "main_id"],
                                                  aliases=["land_occupation_id", "main_id"],
                                                  localize=True),
                    popup=folium.GeoJsonPopup(fields=["land_occupation_id", "main_id"],
                                              aliases=["land_occupation_id", "main_id"],
                                              localize=True)
                ).add_to(m)
            except Exception as e:
                st.warning(f"Could not render Land_occupation layer: {e}")
    else:
        st.info("No facilities selected to show Land_occupation polygons.")

folium.LayerControl().add_to(m)

# Render map (streamlit-folium)
st_map = st_folium(m, width=None, height=650)

# ---------- TABLE ----------
st.subheader(f"Filtered table: {dataset_name} (up to {MAX_PREVIEW_ROWS} rows)")
if dataset_df is None or dataset_df.empty:
    st.warning("No rows match the current filters for this dataset.")
else:
    total_rows = len(dataset_df)
    if total_rows > MAX_PREVIEW_ROWS:
        st.info(f"Dataset has {total_rows:,} rows. Showing first {MAX_PREVIEW_ROWS:,} rows.")
        st.dataframe(dataset_df.head(MAX_PREVIEW_ROWS).reset_index(drop=True), use_container_width=True)
        if st.button("Show ALL rows (may be slow)"):
            st.dataframe(dataset_df.reset_index(drop=True), use_container_width=True)
    else:
        st.dataframe(dataset_df.reset_index(drop=True), use_container_width=True)

    st.download_button("Download filtered table (CSV)",
                       data=dataset_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{dataset_name}_filtered.csv", mime="text/csv")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Marin Pellan© 2025 MetalliCan Data Explorer")
