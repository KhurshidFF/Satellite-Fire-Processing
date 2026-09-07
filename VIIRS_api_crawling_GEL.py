"""
NASA FIRMS(VIIRS) 3종 위성 통합 화재 탐지 및 한반도 마스킹 스크립트

이 스크립트는 NASA FIRMS API를 이용해 VIIRS 계열 3개 위성(SNPP, NOAA-20, NOAA-21)의
화재 탐지 결과를 수집 및 통합하고, 한반도 영역(Shapefile)으로 마스킹하여 저장함.

처리절차
1) 날짜 설정: 실행일 기준 '전일(Yesterday)'을 시작일로 설정하고, 1일간(Day Range=1)의 데이터를 요청.
2) API 호출: 지정한 BBOX 내의 3개 위성(SNPP, NOAA-20, NOAA-21) 화재 데이터를 각각 다운로드.
3) 데이터 통합: 수집된 3개 데이터프레임을 하나로 병합(Merge) 후 원본 CSV 저장.
4) 공간 필터링: 'shp' 폴더의 한반도 경계 파일(Shapefile)을 로드하여 통합 데이터를 클리핑(Masking).
5) 결과 저장: 마스킹된 결과만 'files' 폴더에 GeoJSON 포맷으로 저장.

입력/설정값
- api_key: FIRMS API 키
- products: 대상 위성 리스트 (VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT)
- west, south, east, north: 검색할 사각 영역(BBOX)
- project_root:
    - shp: 마스킹에 사용할 Shapefile 경로 (.../shp)
    - files: 결과물이 저장될 경로 (.../files)

출력
- GeoJSON: {out_dir}/firms_{YYYYMMDD}.geojson
    (※ 주의: 파일명은 KST 날짜를 따르나, GeoJSON 내부에 저장되는 관측 시각 속성은 UTC 기준임)

운영 참고
- 위성 통과 시각: 한국 상공 약 01:30(새벽), 13:30(오후).
- 데이터 지연: 약 3시간 내외.
- 스케줄링: KST 기준 오전/오후 5시에 실행 시, UTC 시차를 고려해 '전일' 데이터부터 조회해야 새벽 패스 데이터를 누락 없이 확보 가능함.
"""

import requests, os, time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import date, timedelta
from io import StringIO

def download_firms_dataframe(api_key, product, west, south, east, north, day_range, target_date, max_retries=3):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{product}/{west},{south},{east},{north}/{day_range}/{target_date}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, verify=False, timeout=60)
            if response.status_code != 200 or "latitude" not in response.text:
                return pd.DataFrame()

            df = pd.read_csv(StringIO(response.text))
            df['source_product'] = product
            print(f"[{product}] Downloaded {len(df)} fires.")
            return df

        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return pd.DataFrame()

def apply_mask_and_save_geojson(df, shapefile_path, out_geojson):
    try:
        if df.empty:
            print("DataFrame is empty. No files created.")
            return

        # 1. Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf.set_crs(epsg=4326, inplace=True) 

        # 2. Load Mask Shapefile
        if os.path.exists(shapefile_path):
            print(f"Loading mask: {shapefile_path}")
            mask_gdf = gpd.read_file(shapefile_path)
            
            if mask_gdf.crs != gdf.crs:
                mask_gdf = mask_gdf.to_crs(gdf.crs)

            # 3. Apply Clip
            initial_count = len(gdf)
            gdf_masked = gpd.clip(gdf, mask_gdf)
            final_count = len(gdf_masked)
            
            print(f">> Mask applied. Points reduced from {initial_count} to {final_count}.")
            
            if final_count == 0:
                print("Warning: All fire points were filtered out by the mask.")
                return

            gdf = gdf_masked
        else:
            print(f"Warning: Shapefile not found at {shapefile_path}. Saving without masking.")

        # 4. Save Output
        gdf.to_file(out_geojson, driver='GeoJSON')
        print(f"Saved GeoJSON: {out_geojson}")

    except Exception as e:
        print(f"Error in processing/masking: {e}")


if __name__ == '__main__' : 
   
    api_key = " " # Transaction limit: 5000 transactions / 10 minutes
    
    # Date Setup: Yesterday + Today (2 days)
    start_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    day_range = 2
    exec_date_str = date.today().strftime('%Y%m%d')

    west, south, east, north = 123.0, 33.0, 132.0, 43.5
    products = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
       
    # Path settings
    project_root = 'C:/Users/User/GEL Dropbox/GEL Team Folder/Projects/산림청_화선탐지/코드공유/Minkyu'
    out_dir = os.path.join(project_root, 'files')
    os.makedirs(out_dir, exist_ok=True)
    shapefile_path = os.path.join(project_root, 'shp', '한반도_4326.shp')

    # Output Filename
    out_geojson = os.path.join(out_dir, f'firms_{exec_date_str}.geojson')
   
    # Download & Process
    all_dfs = []
    print(f"Starting download for period: {start_date} (Range: {day_range} days)")
    
    for product in products:
        df = download_firms_dataframe(api_key, product, west, south, east, north, day_range, start_date)
        if not df.empty:
            all_dfs.append(df)
            
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
             
        # Proceed to mask and save GeoJSON
        apply_mask_and_save_geojson(merged_df, shapefile_path, out_geojson)
    else:
        print("No fires detected from any satellite.")
