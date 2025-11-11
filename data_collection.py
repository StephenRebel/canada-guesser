import requests
import mercantile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import time
from typing import List, Dict
import json
import logging

#Config
TOKEN = os.getenv("MAPILLARY_API_KEY")
if not TOKEN:
    raise RuntimeError("No API token found.")
IMAGES_PER_CITY = 10000
MASTER_DIR = r"/canada_full_city" # Change to donwload target directory
FAILED_LOG_PATH = "failed_tiles.log"
FAILED_DOWNLOAD_PATH = "failed_downloads.jsonl"
DATA_DIR = "collected_files"

# Update as needed for system requirements
MAX_DOWNLOAD_WORKERS = 30
MAX_WORKERS = 30
REQUESTS_PER_SECOND = 10

CITIES = {
    #"ottawa-gatineau":      [-75.938347,45.244301,-75.45704,45.502476],
    #"toronto":              [-79.587052,43.60741,-79.113219,43.825279],
    #"montreal":             [-73.840956,45.413688,-73.487333,45.652261]
    #"vancouver":            [-123.265566,49.198931,-123.023242,49.316171],
    #"calgary":              [-114.279455,50.848748,-113.909353,51.188257],
    #"edmonton":             [-113.713841,53.394321,-113.344344,53.651086],
    # "winnipeg":             [-97.325875,49.763144,-96.956529,49.972925],
    # "saskatoon":            [-106.765098,52.071159,-106.536187,52.202335],
    # "quebec_city":          [-71.371256,46.762893,-71.190277,46.879747],
    # "hamilton":             [-80.045308,43.19374,-79.740724,43.301648],
    # "kitchener-waterloo":   [-80.573458,43.326851,-80.257256,43.506828],
    # "halifax":              [-63.823841,44.53971,-63.205,44.851819],
    # "victoria":             [-123.391097,48.405878,-123.273155,48.472826],
    #  "st_johns":             [-52.919474,47.449785,-52.611841,47.659795],
    "charlottetown":        [-64.4315,45.9687,-61.9573,47.0479]
}

random.seed(42) # Reproducible seeding

# Logging of errors, thread safe
logger = logging.getLogger("failed_tiles")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(FAILED_LOG_PATH, encoding="utf-8", mode="a")
logger.addHandler(fh)

# Rate limiter class to not overwhelm API
class RateLimiter:
    def __init__(self, calls_per_second):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_called = 0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_called
        sleep_time = max(0, self.min_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_called = time.time()

rate_limiter = RateLimiter(REQUESTS_PER_SECOND)

# Scraping based on map tiles. Log any that fail completely for later
def log_failed_tile_obj(tile: mercantile.Tile, city: str, error: str):
    entry = {
        "city": city,
        "x": tile.x,
        "y": tile.y,
        "z": tile.z,
        "error": error,
        "timestamp": time.time()
    }
    # logging is thread-safe, write JSONL
    logger.info(json.dumps(entry))

# Used for saving all scraped image IDs and URL (can sample more data without rescraping)
def write_jsonl(path: str, objs: List[Dict]):
    # Write JSONL to file.
    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            json_line = json.dumps(obj, ensure_ascii=False)
            f.write(json_line + "\n")

# Thread actor function to collect all images (just URL and IDs) per tile based on mercantile for map tiling
def fetch_images_from_tile(tile: mercantile.Tile, city: str, idx: int, total: int, retries: int = 3) -> list[Dict]:
    rate_limiter.wait()

    # Format mercantile bounds to Mapillary API string
    west, south, east, north = mercantile.bounds(tile)
    bbox_str = f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"

    # A few retries as the API tends to have some failures but rarely more than a couple spread across all tiles
    for attempt in range(retries):
        resp = None

        try:
            resp = requests.get(
                "https://graph.mapillary.com/images",
                params={
                    "access_token": TOKEN,
                    "bbox": bbox_str,
                    "fields": "id,thumb_1024_url,geometry,captured_at,camera_type", # Change depending on image size and other data
                    "limit": 2000
                },
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            
            # Collect returned images
            filtered = []
            for img in data:
                cam_type = img.get("camera_type", "").lower()
                if cam_type in ["spherical", "equirectangular"]: # NOTE If don't want the panaramics use this from API Documentation
                    continue
                img["city"] = city
                filtered.append(img)
            return filtered

        # Catching server side errors
        except requests.exceptions.HTTPError as e:
            code = resp.status_code if resp is not None else "N/A"
            if code >= 500 and attempt < retries - 1:
                wait = 2 ** attempt + random.random() # Retrying with jitter added in case
                print(f"Retrying {city}: tile {idx + 1}/{total} ({tile.x}/{tile.y}/{tile.z}). HTTP {code}. retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            log_failed_tile_obj(tile, city, f"HTTP {code}: {e}")
            print(f"Failure {city}: tile {idx + 1}/{total} ({tile.x}/{tile.y}/{tile.z}). HTTP {code}. {e}")
            return []

        # General errors
        except Exception as e:
            log_failed_tile_obj(tile, city, f"{type(e).__name__}: {e}")
            print(f"Failure {city}: tile {idx + 1}/{total} ({tile.x}/{tile.y}/{tile.z}). {e}")
            return []

#Fetch images
all_images: List[Dict] = []
seen_ids: set[str] = set() 

os.makedirs(DATA_DIR, exist_ok=True)

print("Collecting images. Tile fetching from Mapillary API...")
start_all = time.time()

# Driver code for tiling and dispatching jobs
for city, bbox in CITIES.items():
    start_city = time.time()
    print(f"\nTiling {city.title()}, zoom 14")

    tiles = list(mercantile.tiles(*bbox, zooms=[14]))
    print(f"  Total tiles: {len(tiles)}. Using {MAX_WORKERS} threads.")

    city_images: List[Dict] = []
    city_seen: set[str] = set()

    # Main fetch block
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_tile = {
            executor.submit(fetch_images_from_tile, tile, city, idx, len(tiles)): tile
            for idx, tile in enumerate(tiles)
        }

        # Tqdm for real time progress, and collection results
        for future in tqdm(as_completed(future_to_tile), total=len(tiles), desc=f"{city} tiles", leave=False):
            imgs = future.result()

            for img in imgs:
                img_id = img["id"]
                if img_id not in seen_ids and img_id not in city_seen:
                    city_images.append(img)
                    city_seen.add(img_id)
                    seen_ids.add(img_id)

    city_path = os.path.join(DATA_DIR, f"{city}.jsonl")
    write_jsonl(city_path, city_images)

    end_city = time.time()
    print(f"{city.title()}: {len(city_images)} unique images ({len(city_seen)} total fetched) in {end_city - start_city:.1f}s")
    all_images.extend(city_images)

end_all = time.time()
print(f"\nCollected {len(all_images)} unique images across all cities in {end_all - start_all:.1f}s")

# Only select specific number of images to download
images_to_download = []

for city in CITIES.keys():
    # Filter out images with no 1024px valid URL
    city_imgs = [
        img for img in all_images
        if img["city"] == city
        and img.get("thumb_1024_url")
        and isinstance(img["thumb_1024_url"], str)
        and img["thumb_1024_url"].startswith("http")
    ]
    
    # If not enough take all, otherwise random sample
    if len(city_imgs) <= IMAGES_PER_CITY:
        selected = city_imgs
    else:
        selected = random.sample(city_imgs, IMAGES_PER_CITY)
    
    images_to_download.extend(selected)

print(f"\nSelected {len(images_to_download)} images to download ({IMAGES_PER_CITY} per city.")

#Download images to computer
os.makedirs(MASTER_DIR, exist_ok=True)
total_size = 0
failed_downloads = []

# Download an image. Returns: success, fail_record or None
def download(img):
    global total_size # For relative estimates, might not explicitly be thread safe
    city, img_id = img["city"], img["id"]
    url = img.get("thumb_1024_url") # Change depending on image we want to donwload
    path = os.path.join(MASTER_DIR, city, f"{img_id}.jpg")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Skip if already exists
    if os.path.exists(path):
        total_size += os.path.getsize(path)
        return True, None

    # No retry as downloads tended to succeed for valid URL, failed saved in case.
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.content
        with open(path, "wb") as f:
            f.write(data)
        total_size += len(data)
        return True, None

    except Exception as e:
        # store enough info to retry later
        return False, {
            "city": city,
            "id": img_id,
            "url": url,
            "error": str(e),
            "timestamp": time.time()
        }

print("\nDownloading images...")
start_dl = time.time()

# Again threaded otherwise will take long time
with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
    futures = [pool.submit(download, img) for img in images_to_download]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="img"):
        ok, fail_record = future.result()
        if not ok:
            failed_downloads.append(fail_record)

end_dl = time.time()

# Save any failed downloads to a file
if failed_downloads:
    with open(os.path.join(DATA_DIR, FAILED_DOWNLOAD_PATH), 'a') as f:
        for obj in failed_downloads:
            json_line = json.dumps(obj, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"\n{len(failed_downloads)} downloads failed, written to {FAILED_DOWNLOAD_PATH}")
else:
    print(f"\nAll downloads have succeeded")

print(f"\nDONE! {len(images_to_download)} images. {total_size / (1024**3):.2f} GB")
print(f"Download time: {end_dl - start_dl:.1f}s")
