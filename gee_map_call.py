import ee
import time
import os
from tqdm import tqdm
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


# ---- Earth Engine Initialization ----
def initialize_earth_engine(project_id):
    try:
        ee.Initialize(project=project_id)
    except ee.EEException:
        print("Authenticating Earth Engine for the first time...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print("Earth Engine initialized.")


# ---- Google Drive Authentication ----
def authenticate_drive():
    gauth = GoogleAuth()

    # Try loading saved credentials
    gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.credentials is None:
        print("Authenticating Google Drive for the first time...")
        gauth.LocalWebserverAuth()  # Interactive login
    elif gauth.access_token_expired:
        print("Refreshing expired Google Drive credentials...")
        gauth.Refresh()
    else:
        gauth.Authorize()

    # Save credentials for next time
    gauth.SaveCredentialsFile("mycreds.txt")

    return GoogleDrive(gauth)


# ---- Export and Download Function ----
def export_maps(min_lon, min_lat, max_lon, max_lat, year='2023'):
    roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    dem = ee.Image("USGS/SRTMGL1_003").select('elevation')
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

    exports = {
        'DEM': dem,
        'SR_B3': b3,
        'SR_B4': b4,
        'SR_B5': b5
    }

    tasks = []
    for name, image in exports.items():
        file_prefix = name.lower()
        task = ee.batch.Export.image.toDrive(
            image=image.clip(roi),
            description=name,
            folder='GEE_exports',
            fileNamePrefix=file_prefix,
            scale=30,
            region=roi.getInfo()['coordinates'],
            maxPixels=1e13
        )
        task.start()
        print(f"Started export task: {name}")
        tasks.append((name, task))

    # Monitor tasks with tqdm
    print("\n📦 Monitoring Earth Engine export progress...\n")
    with tqdm(total=len(tasks), desc="GEE Export Progress") as pbar:
        completed = [False] * len(tasks)

        while not all(completed):
            time.sleep(10)
            for i, (name, task) in enumerate(tasks):
                if not completed[i]:
                    status = task.status()
                    if status['state'] == 'COMPLETED':
                        completed[i] = True
                        pbar.update(1)
                        print(f"✅ {name} export completed.")
                    elif status['state'] == 'FAILED':
                        completed[i] = True
                        pbar.update(1)
                        print(f"❌ {name} export failed: {status.get('error_message', 'Unknown error')}")

    # Ensure local folder exists
    os.makedirs('maps', exist_ok=True)

    # Authenticate Google Drive
    drive = authenticate_drive()

    # Locate the 'GEE_exports' folder ID
    print("\n🔍 Locating 'GEE_exports' folder in Google Drive...\n")
    folder_list = drive.ListFile({'q': "mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
    folder_id = None
    for folder in folder_list:
        if folder['title'] == 'GEE_exports':
            folder_id = folder['id']
            break

    if not folder_id:
        raise Exception("❌ 'GEE_exports' folder not found in your Drive.")

    # Get list of exported files in the folder
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    # Filter relevant files
    download_files = [
        file for file in file_list
        if file['title'].startswith(('dem', 'sr_b3', 'sr_b4', 'sr_b5')) and file['title'].endswith('.tif')
    ]

    # Download with progress bar
    print("\n⬇️ Downloading exported .tif files to 'maps/' folder...\n")
    with tqdm(total=len(download_files), desc="Drive Download Progress") as pbar:
        for file in download_files:
            filename = file['title']
            print(f"Downloading {filename}...")
            file.GetContentFile(f'maps/{filename}')
            pbar.update(1)

    print("\n✅ All exports successfully downloaded to 'maps/' folder.")


if __name__ == "__main__":
    PROJECT_ID = "terraintel-460116"
    initialize_earth_engine(PROJECT_ID)
    export_maps(86.5, 27.5, 87.0, 28.0)
