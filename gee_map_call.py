import ee
import gee_map_call

# Initialize GEE
ee.Initialize(project="terraintel-460116")

# Define function to fetch and export data
def export_maps(min_lon, min_lat, max_lon, max_lat, year='2023'):
    roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    
    # DEM Layer
    dem = ee.Image("USGS/SRTMGL1_003").select('elevation')
    
    # Landsat 9 surface reflectance median composite
    landsat = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filterBounds(roi)
        .filterDate(f'{year}-01-01', f'{year}-12-31')
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
        .median()
    )
    
    b3 = landsat.select('SR_B3')
    b4 = landsat.select('SR_B4')
    b5 = landsat.select('SR_B5')

    # List of images to export
    exports = {
        'DEM_SRTM': dem,
        'SR_B3': b3,
        'SR_B4': b4,
        'SR_B5': b5
    }

    for name, image in exports.items():
        task = ee.batch.Export.image.toDrive(
            image=image.clip(roi),
            description=name,
            folder='GEE_exports',
            fileNamePrefix=name.lower(),
            scale=30,
            region=roi.getInfo()['coordinates'],
            maxPixels=1e13
        )
        task.start()
        print(f"Export task started: {name}")

# Example: Nepal region
export_maps(86.5, 27.5, 87.0, 28.0)
