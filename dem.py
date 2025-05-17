import rasterio
import numpy as np
from rasterio.transform import xy
from rasterio.enums import Resampling
from scipy.ndimage import sobel
import json

dem_file = "dem.TIF"
band3_file = "SR_B3.TIF"  # Green
band4_file = "SR_B4.TIF"  # Red
band5_file = "SR_B5.TIF"  # NIR
output_file = "terrain_data.jsonl"  # Output file


#Compute slope and aspect from DEM
def compute_slope_aspect(dem_array):
    dx = sobel(dem_array, axis=1, mode='constant') / 8.0
    dy = sobel(dem_array, axis=0, mode='constant')
    slope = np.arctan(np.hypot(dx, dy)) * (180 / np.pi)
    aspect = np.arctan2(-dx, dy)
    aspect = np.degrees(aspect)
    aspect = np.where(aspect < 0, 90.0 - aspect, 360.0 - aspect + 90.0)
    return slope, aspect


#Load DEM and calculate slope & aspect
with rasterio.open(dem_file) as dem_src:
    dem = dem_src.read(1, resampling=Resampling.bilinear)
    transform = dem_src.transform
    nodata = dem_src.nodata if dem_src.nodata is not None else -9999
    dem = np.where(dem == nodata, np.nan, dem)
    slope, aspect = compute_slope_aspect(dem)

#Load Landsat Surface Reflectance bands
with rasterio.open(band3_file) as b3, \
        rasterio.open(band4_file) as b4, \
        rasterio.open(band5_file) as b5:
    green = b3.read(1).astype("float32")
    red = b4.read(1).astype("float32")
    nir = b5.read(1).astype("float32")

#Calculate NDVI and NDWI
np.seterr(divide='ignore', invalid='ignore')
ndvi = (nir - red) / (nir + red)
ndwi = (green - nir) / (green + nir)

#Export data to JSONL
step = 10  # controls sampling density

with open(output_file, "w") as f:
    for row in range(0, dem.shape[0], step):
        for col in range(0, dem.shape[1], step):
            elev = dem[row, col]
            if np.isnan(elev): continue

            lon, lat = xy(transform, row, col)
            data = {
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "elevation": float(elev),
                "slope": float(slope[row, col]),
                "aspect": float(aspect[row, col]),
                "ndvi": float(ndvi[row, col]) if np.isfinite(ndvi[row, col]) else None,
                "ndwi": float(ndwi[row, col]) if np.isfinite(ndwi[row, col]) else None
            }
            f.write(json.dumps(data) + "\n")

print(f"Done. Data written to {output_file}")
