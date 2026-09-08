import requests, os, time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from io import StringIO

def download_firms_dataframe(api_key, product, west, south, east, north, day_range, target_date, max_retries=3):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{product}/{west},{south},{east},{north}/{day_range}/{target_date}"
    
    for attempt in range(max_retries):
        try:
            # SSL xatoligini chetlab o'tish uchun verify=False (NASA API ba'zan talab qiladi)
            response = requests.get(url, verify=False, timeout=60)
            if response.status_code != 200 or "latitude" not in response.text:
                return pd.DataFrame()

            df = pd.read_csv(StringIO(response.text))
            df['source_product'] = product
            print(f"[{product}] {target_date} dan boshlab {day_range} kun ichida {len(df)} ta nuqta topildi.")
            return df

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print(f"Xato yuz berdi [{product}]: {e}")
                return pd.DataFrame()

def apply_mask_and_export(df, shapefile_path, out_csv, out_geojson):
    try:
        if df.empty:
            print("DataFrame bo'sh. Fayllar yaratilmadi.")
            return

        # 1. Tablyutsani spatial nuqtalarga o'tkazish
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf.set_crs(epsg=4326, inplace=True) 

        # 2. Shapefile (niqob) yuklash
        if os.path.exists(shapefile_path):
            mask_gdf = gpd.read_file(shapefile_path)
            if mask_gdf.crs != gdf.crs:
                mask_gdf = mask_gdf.to_crs(gdf.crs)

            # 3. Spatial Clip (faqat hudud ichidagilarni qoldirish)
            initial_count = len(gdf)
            gdf_masked = gpd.clip(gdf, mask_gdf)
            final_count = len(gdf_masked)
            
            print(f">> Mask qo'llanildi. Nuqtalar soni: {initial_count} -> {final_count}")
            
            if final_count == 0:
                print("Ogohlantirish: Barcha nuqtalar maskadan tashqarida qoldi.")
                return
            gdf = gdf_masked
        else:
            print(f"Ogohlantirish: Shapefile topilmadi ({shapefile_path}).")

        # 4. Export CSV
        clean_df = pd.DataFrame(gdf.drop(columns='geometry'))
        clean_df.to_csv(out_csv, index=False)
        print(f"Saqlandi (CSV): {out_csv}")

        # 5. Export GeoJSON
        gdf.to_file(out_geojson, driver='GeoJSON')
        print(f"Saqlandi (GeoJSON): {out_geojson}")

    except Exception as e:
        print(f"Qayta ishlashda xato: {e}")


if __name__ == '__main__': 
   
    api_key = "a78b23d832dab9f73e6d169430d26a40" 
    
    # Hamyang yong'ini: 2.21 (21:14) - 2.24 (06:43)
    # 21-fevraldan boshlab 4 kunlik ma'lumotni olamiz
    target_date = '2026-01-10'
    day_range = 1
    exec_date_str = "fffffff_20260110"

    # Koreya hududi koordinatalari
    west, south, east, north = 126.0, 33.1, 129.9, 38.6
    
    # BARCHA VIIRS MAHSULOTLARI (SNPP, NOAA-20, NOAA-21)
    products = [
        "VIIRS_SNPP_NRT", 
        "VIIRS_NOAA20_NRT", 
        "VIIRS_NOAA21_NRT"
    ]
        
    out_dir = '/home/xurshedjonff/VIIRS_HAMYANG_FIRE_OUTPUT/2026/052_055/'
    shapefile_path = '/home/xurshedjonff/KOREA_SHP/HangJeongDong_ver20241018.shp'
    
    os.makedirs(out_dir, exist_ok=True)

    out_geojson = os.path.join(out_dir, f'firms_Ipcheonri_all_viirs_{exec_date_str}.geojson')
    out_csv = os.path.join(out_dir, f'firms_Ipcheonri_all_viirs_{exec_date_str}.csv') 
   
    all_dfs = []
    print(f"Hamyang yong'ini uchun yuklash boshlandi: {target_date} (Range: {day_range} kun)")
    
    for product in products:
        df = download_firms_dataframe(api_key, product, west, south, east, north, day_range, target_date)
        if not df.empty:
            all_dfs.append(df)
            
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Tungi o'tish vaqtlarini filtrlaymiz (16:00 - 18:00 UTC)
        merged_df['acq_time'] = merged_df['acq_time'].astype(int)
        time_filtered_df = merged_df[(merged_df['acq_time'] >= 1600) & (merged_df['acq_time'] <= 1800)]
        
        print(f"Tungi o'tishlar filtri qo'llanildi. {len(time_filtered_df)} ta validation nuqtasi qoldi.")
              
        if not time_filtered_df.empty:
            apply_mask_and_export(time_filtered_df, shapefile_path, out_csv, out_geojson)
        else:
            print("Belgilangan vaqt oralig'ida tungi o'tishlar topilmadi.")
    else:
        print("Hech qaysi yo'ldoshdan ma'lumot yuklanmadi.")