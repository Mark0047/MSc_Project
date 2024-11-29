import os
import json
import asyncio
import aiohttp
from tqdm import tqdm

# CKAN API URLs
CKAN_PACKAGE_LIST_URL = "https://ckan.publishing.service.gov.uk/api/action/package_list"
CKAN_PACKAGE_SHOW_URL = "https://ckan.publishing.service.gov.uk/api/action/package_show?id="

# Directory to save JSON files
SAVE_DIR = "gov_data_json"
os.makedirs(SAVE_DIR, exist_ok=True)

# Parameters
MAX_CONCURRENT_REQUESTS = 10  # Limit to prevent overloading the server

async def fetch_all_package_ids():
    """Fetch all document IDs from CKAN asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(CKAN_PACKAGE_LIST_URL) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('result', [])
            else:
                raise Exception(f"Error fetching package list: {response.status}")

async def fetch_and_save_json(session, package_id):
    """Fetch the JSON metadata for a single package and save it locally."""
    try:
        async with session.get(CKAN_PACKAGE_SHOW_URL + package_id) as response:
            if response.status == 200:
                data = await response.json()
                save_path = os.path.join(SAVE_DIR, f"{package_id}.json")
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=4)
                return True
            else:
                print(f"Failed to fetch data for package {package_id} with status {response.status}")
                return False
    except Exception as e:
        print(f"Error fetching data for package {package_id}: {e}")
        return False

async def download_all_packages():
    """Download JSON data for all packages with concurrency and error handling."""
    package_ids = await fetch_all_package_ids()
    
    # Set up a semaphore to limit the number of concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def safe_fetch(package_id):
        """Fetch with semaphore control and retry mechanism."""
        async with semaphore:
            retries = 3
            for attempt in range(retries):
                success = await fetch_and_save_json(session, package_id)
                if success:
                    break
                elif attempt < retries - 1:
                    print(f"Retrying package {package_id}... ({attempt+1}/{retries})")
                    await asyncio.sleep(1)  # Wait before retrying
            return success

    # Download packages concurrently with progress bar
    async with aiohttp.ClientSession() as session:
        tasks = [safe_fetch(package_id) for package_id in package_ids]
        for future in tqdm(asyncio.as_completed(tasks), total=len(package_ids), desc="Downloading JSONs"):
            await future  # Wait for each download task to complete

def main():
    # Run the asynchronous download process
    asyncio.run(download_all_packages())
    print("All JSON files downloaded.")

if __name__ == "__main__":
    main()
