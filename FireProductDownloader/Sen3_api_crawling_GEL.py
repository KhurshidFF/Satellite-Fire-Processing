import os, requests, s3fs, io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta

# ================= CONFIGURATION =================
CLIENT_ID, CLIENT_SECRET = " ", " "
S3_ACCESS_KEY, S3_SECRET_KEY = " ", " "
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
SHP_PATH = "/home/xurshedjonff/files/SouthKorea_Shp/gadm41_KOR_0.shp"
OUT_DIR = "/home/xurshedjonff/files/S3_FRP_NRT/"
DAY_RANGE = 2 
# =================================================

def get_token():
    r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", 
                      data={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "grant_type": "client_credentials"})
    return r.json()['access_token']

def process_single_csv(fs, s3_path, band_label):
   
    try:
        if not fs.exists(s3_path):
            return pd.DataFrame()

        with fs.open(s3_path, mode='rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            lines = content.splitlines()
            header_idx = next((i for i, l in enumerate(lines) if 'lat(deg)' in l), None)
            
            if header_idx is None:
                return pd.DataFrame()

            # Read Data
            df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), sep=None, engine='python')
            df.columns = df.columns.str.strip()
            df.rename(columns={'FRP (MW)': 'FRP(MW)'}, inplace=True)

            # Filter: FRP > 0
            df['FRP(MW)'] = pd.to_numeric(df['FRP(MW)'], errors='coerce')
            df = df[df['FRP(MW)'] > 0].dropna(subset=['lat(deg)', 'lon(deg)'])
            
            if not df.empty:
                # Add band_type
                df['band_type'] = band_label 
                return df
            
    except Exception as e: print(f"Error processing")
    
    return pd.DataFrame()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sk_gdf = gpd.read_file(SHP_PATH).to_crs(epsg=4326)
    token = get_token()
    fs = s3fs.S3FileSystem(key=S3_ACCESS_KEY, secret=S3_SECRET_KEY, endpoint_url=S3_ENDPOINT)
    
    start_time = (datetime.utcnow() - timedelta(days=DAY_RANGE-1)).replace(hour=0, minute=0, second=0).strftime('%Y-%m-%dT%H:%M:%SZ')
    b = sk_gdf.total_bounds
    aoi = f"POLYGON(({b[0]} {b[1]}, {b[2]} {b[1]}, {b[2]} {b[3]}, {b[0]} {b[3]}, {b[0]} {b[1]}))"
    
    query = (f"Collection/Name eq 'SENTINEL-3' and contains(Name, 'SL_2_FRP') and "
             f"ContentDate/Start ge {start_time} and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}')")

    # Sentinel-3A/B Day/Night passes request from website
    r = requests.get("https://catalogue.dataspace.copernicus.eu/odata/v1/Products", 
                     headers={"Authorization": f"Bearer {token}"}, 
                     params={"$filter": query, "$orderby": "ContentDate/Start desc", "$top": 2})
    products = r.json().get('value', [])
    
    for prod in products:
        full_name = prod['Name'] if prod['Name'].endswith('.SEN3') else f"{prod['Name']}.SEN3"
        utc_ts = full_name.split('_')[7]
        yyyy, mm, dd = utc_ts[:4], utc_ts[4:6], utc_ts[6:8]
        target_geojson = os.path.join(OUT_DIR, f"Sen3_FRP_{utc_ts}.geojson")
        if os.path.exists(target_geojson): continue
        base_path = f"eodata/Sentinel-3/SLSTR/SL_2_FRP___/{yyyy}/{mm}/{dd}/{full_name}"

        # Processing both individually
        df_mwir = process_single_csv(fs, f"{base_path}/FRP_MWIR1km_standard.csv", "MWIR_1km")
        df_swir = process_single_csv(fs, f"{base_path}/FRP_SWIR500m.csv", "SWIR_500m")
        # Merge
        combined_df = pd.concat([df_mwir, df_swir], ignore_index=True)
        if not combined_df.empty:
            gdf = gpd.GeoDataFrame(combined_df, geometry=[Point(xy) for xy in zip(combined_df['lon(deg)'], combined_df['lat(deg)'])], crs="EPSG:4326")
            gdf_masked = gpd.clip(gdf, sk_gdf).copy()
            
            if not gdf_masked.empty:
                gdf_masked.loc[:, 'fire_datetime_utc'] = utc_ts
                
                # Geojson columns for merging information
                cols_to_keep = ['FRP(MW)', 'band_type', 'fire_datetime_utc', 'geometry']
                gdf_masked[cols_to_keep].to_file(target_geojson, driver='GeoJSON')
                print(f"[SUCCESS] Saved {len(gdf_masked)} fires")
    print("Done.")

if __name__ == "__main__":
    main()