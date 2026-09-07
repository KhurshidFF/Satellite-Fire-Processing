"""
GK2A(천리안 2A) 산불탐지(FF) 산출물 GeoJson 변환 스크립트

이 스크립트는 기상청 APIHub에서 GK2A 위성의 산불(FF) 산출물을 다운로드하여
화재 픽셀을 추출하고, 한반도 형상(Shapefile)으로 마스킹한 뒤 결과물을 저장함.

처리절차
1) 시간 산정: 시스템 시각(KST 가정)에서 지연시간(20분)을 뺀 '대상 시각'을 산출.
2) 요청 및 다운로드: 대상 시각을 UTC로 변환하여 APIHub에 데이터 요청 및 다운로드(.gb2).
3) 좌표 변환: FF=1(화재) 픽셀 추출 후 LCC 좌표계를 위경도(WGS84)로 변환.
4) 공간 마스킹: 변환된 화점 데이터를 Shapefile 경계로 클리핑(Clip).
5) 결과 저장: 마스킹 된 화점 데이터를 KST 시각이 적용된 파일명으로 저장하고 원본 파일 삭제.

입력/설정값
- api_key: 기상청 APIHub 인증키
- project_root:
    - shp: 마스킹에 사용할 Shapefile 경로 (.../shp)
    - files: 결과물이 저장될 경로 (.../files)
- latency: 데이터 수집 지연시간 (기본 20분 설정)

출력(파일명은 KST 기준)
- GeoJSON: {out_dir}/gk2a_kst_{YYYYMMDDHHMM}.geojson

운영 참고
- 본 코드는 시스템 시간을 KST로 가정하고 동작.
- GK2A 산출물은 10분 단위로 생성되며, 수집 안정성을 위해 20분의 지연시간(Latency)을 두고 실행하는 것을 권장.
"""

import os, requests, time
from datetime import datetime, timedelta
import h5py
import numpy as np
import pandas as pd             # Added
import geopandas as gpd         # Added
from shapely.geometry import Point # Added
from pyproj import Proj

def download_GK2A_FF(filename, api_key, tmfc, max_retries=3):
    URL = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE2/FF/KO/data"

    for attempt in range(max_retries):
        try:
            response = requests.get(URL, params={"authKey": api_key, "date": tmfc}, timeout=60)
            response.raise_for_status()

            with open(filename, "wb") as f:
                f.write(response.content)
            return 

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise

def process_fires_and_mask(temp_file, shapefile_path, out_geojson):
    try:
        with h5py.File(temp_file, 'r') as f:
            # Read data
            if 'FF' not in f.keys():
                print("Error: 'FF' dataset not found in file.")
                return

            ff = f['FF'][:]
            dqf = f['DQF_FF'][:]
            
            # 2. Get projection metadata 
            attrs = {} 
            lat_1 = float(attrs.get('standard_parallel1', 30.0))
            lat_2 = float(attrs.get('standard_parallel2', 60.0))
            lat_0 = float(attrs.get('origin_latitude', 38.0))
            lon_0 = float(attrs.get('central_meridian', 126.0))
            ul_x  = float(attrs.get('upper_left_easting', -899000.0))
            ul_y  = float(attrs.get('upper_left_northing', 899000.0))
            px_size = float(attrs.get('pixel_size', 2000.0))

            # Find fire pixels 
            rows, cols = np.where(ff == 1)
            
            if len(rows) == 0:
                print("No fires detected (FF=1 not found).")
                return
            
            print(f"Detected {len(rows)} raw fire pixels. Calculating coordinates...")

            # Convert to lat/lon
            half_px = px_size / 2.0 
            x_m = ul_x + (cols * px_size) + half_px
            y_m = ul_y - (rows * px_size) - half_px
            
            # Setup Projection (LCC -> WGS84)
            p = Proj(proj='lcc', lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, 
                     lon_0=lon_0, a=6378137.0, rf=298.25722)
            lons, lats = p(x_m, y_m, inverse=True)
            
            # Get corresponding DQF values
            valid_dqf = dqf[rows, cols]

            # Create GeoDataFrame
            df = pd.DataFrame({
                'FF': 1,
                'DQF': valid_dqf,
                'longitude': lons,
                'latitude': lats
            })
            
            geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
            gdf.set_crs(epsg=4326, inplace=True)

            # Apply Shapefile Mask
            if os.path.exists(shapefile_path):
                print(f"Loading mask: {shapefile_path}")
                mask_gdf = gpd.read_file(shapefile_path)
                
                if mask_gdf.crs != gdf.crs:
                    mask_gdf = mask_gdf.to_crs(gdf.crs)
                
                # Clip
                initial_count = len(gdf)
                gdf_masked = gpd.clip(gdf, mask_gdf)
                final_count = len(gdf_masked)
                
                print(f">> Mask applied. Points reduced from {initial_count} to {final_count}.")
                
                if final_count == 0:
                    print("Warning: All points were outside the mask.")
                    return
                
                gdf = gdf_masked
            else:
                print(f"Warning: Shapefile not found at {shapefile_path}. Skipping mask.")

            # Save output
            gdf.to_file(out_geojson, driver='GeoJSON')
            print(f"Saved GeoJSON: {out_geojson}")

    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == '__main__' : 
   
    api_key = " "
    
    # Path settingszhemdp
    project_root = 'C:/Users/User/GEL Dropbox/GEL Team Folder/Projects/산림청_화선탐지/코드공유/Minkyu'
    out_dir = os.path.join(project_root, 'files')
    os.makedirs(out_dir, exist_ok=True)
    shapefile_path = os.path.join(project_root, 'shp', '한반도_4326.shp') 
    
    # Time Setup
    time_target = datetime.now() - timedelta(minutes=20) # 20min latency
    tmfc_kst = time_target.replace(minute=time_target.minute // 10 * 10).strftime('%Y%m%d%H%M')
    time_utc = time_target - timedelta(hours=9)
    tmfc_utc = time_utc.replace(minute=time_utc.minute // 10 * 10).strftime('%Y%m%d%H%M')
    
    print(f"Target Data Time -> API (UTC): {tmfc_utc} | Filename (KST): {tmfc_kst}")
    
    # Filenames
    temp_file = os.path.join(out_dir, f'gk2a_utc_{tmfc_utc}.gb2')
    out_geojson = os.path.join(out_dir, f'gk2a_kst_{tmfc_kst}.geojson')
        
    # Download & Process
    download_GK2A_FF(temp_file, api_key, tmfc_utc)
            
    if os.path.exists(temp_file):
        process_fires_and_mask(temp_file, shapefile_path, out_geojson)
        os.remove(temp_file)
        print(f"Deleted temp file: {temp_file}")
    else:
        print("Download failed, skipping processing.")
