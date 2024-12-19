import cloudscraper
import requests
from bs4 import BeautifulSoup
import json
import os
import time
from PIL import Image
from dotenv import load_dotenv
import urllib.parse

# Load environment variables from .env file
load_dotenv()

# Global variables for paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Dit is de map 'app'

# De paden voor datasets en afbeeldingen worden relatief vanaf de 'app' directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets', 'pornstar_images')
JSON_PATH = os.path.join(BASE_DIR, 'datasets', 'performers_data.json')

# Load API key from environment variable
API_KEY = os.getenv("THEPORNDB_API_KEY")

def get_theporndb_details(performer_name, page=1, per_page=10, retries=5, delay=4):
    encoded_name = urllib.parse.quote(performer_name)
    encoded_name = encoded_name.replace("_", " ")
    search_url = f"https://api.theporndb.net/performers?q={encoded_name}&page={page}&per_page={per_page}"
    print(f"Searching ThePornDB for: {search_url}")

    headers = {
        "accept": "application/json",
       "Authorization": f"Bearer {API_KEY}"
    }

    for attempt in range(retries):
        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if not data or 'data' not in data or not data['data']:
                print(f"Performer not found on ThePornDB: {performer_name}")
                return None

            performer_data = data['data'][0]
            # Verzamel alle beschikbare afbeeldings-URL's van posters
            posters = performer_data.get('posters', [])
            image_urls = [poster['url'] for poster in posters] if posters else []

            performer_details = {
                'id': performer_data.get('id', ''),
                'slug': performer_data.get('slug', ''),
                'name': performer_data.get('name', ''),
                'bio': performer_data.get('bio', ''),
                'rating': performer_data.get('rating', ''),
                'gender': performer_data['extras'].get('gender', ''),
                'birthday': performer_data['extras'].get('birthday', ''),
                'birthplace': performer_data['extras'].get('birthplace', ''),
                'ethnicity': performer_data['extras'].get('ethnicity', ''),
                'nationality': performer_data['extras'].get('nationality', ''),
                'hair_color': performer_data['extras'].get('hair_colour', ''),
                'eye_color': performer_data['extras'].get('eye_colour', ''),
                'height': performer_data['extras'].get('height', ''),
                'weight': performer_data['extras'].get('weight', ''),
                'measurements': performer_data['extras'].get('measurements', ''),
                'cup_size': performer_data['extras'].get('cupsize', ''),
                'tattoos': performer_data['extras'].get('tattoos', ''),
                'piercings': performer_data['extras'].get('piercings', ''),
                'image_urls': image_urls,  # Voeg alle afbeeldings-URL's toe
                'links': performer_data['extras'].get('links', {})
            }
            return performer_details

        except requests.exceptions.RequestException as e:
            print(f"Error retrieving ThePornDB details for {performer_name}: {e}")
            if attempt + 1 < retries:
                print(f"Retrying {attempt + 1} out of {retries}...") 
                time.sleep(delay)
            else:
                return None

def get_pornhub_performers(max_performers=15000000):
    performers = []
    base_url = "https://nl.pornhub.com/pornstars?performerType=pornstar&page="
    scraper = cloudscraper.create_scraper()

    # Update de User-Agent naar een andere versie om de 403 te vermijden
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }

    page = 1
    while len(performers) < max_performers:
        print(f"--------------------------------------------------------")
        print(f"pagina {page} / ? - performer ? / ?")
        url = f"{base_url}{page}"
        print(f"Scraping page {page}: {url}")
        try:
            response = scraper.get(url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        performer_cards = soup.find_all('li', class_='pornstarLi performerCard')

        if not performer_cards:
            html_snippet = soup.prettify()[:200]
            if "No performer cards found" in html_snippet or "<!DOCTYPE html>" in html_snippet:
                print("No performer cards found. Here is a snippet of the HTML: ")
                print(html_snippet)
                break

        page_data = []

        for card in performer_cards:
            name_tag = card.find('span', class_='pornStarName performerCardName')

            if name_tag:
                name = name_tag.get_text()
                formatted_name = "_".join(name.split())

                # Haal ThePornDB-gegevens op en voeg toe als beschikbaar
                print(f"start zoeken in theporndb voor: {formatted_name}")
                theporndb_details = get_theporndb_details(formatted_name)
                if theporndb_details:
                    print(f"performer {formatted_name} successvol gevonden op pornhub")
                    performer_data = theporndb_details

                    # Voeg performer toe aan lijst als niet dubbel
                    if any(existing.get('theporndb_name') == formatted_name for existing in performers):
                        print(f"Skipped duplicate performer: {formatted_name}")
                        continue

                    performers.append(performer_data)
                    page_data.append(performer_data)
                    print(f"page: {page} | Added performer: {formatted_name}")

                    # Download en optimaliseer afbeelding van ThePornDB
                    image_urls = performer_data.get('image_urls', [])
                    if image_urls:
                        save_path_base = os.path.join(OUTPUT_DIR, formatted_name)
                        downloaded_image_paths = download_and_optimize_images(image_urls, save_path_base, performer_data)

                        # Update image_urls in the performer data met lokale paden
                        performer_data['image_urls'] = downloaded_image_paths

                    # Write the performer data to the JSON file immediately after adding
                    write_to_json(performer_data)

        if len(performers) >= max_performers:
            break

        page += 1
        time.sleep(2)

    return performers

def download_and_optimize_images(urls, save_path_base, performer_data):
    if not urls:
        print(f"No images to download for {performer_data['name']}")
        return []

    # Maak de map voor de performer als deze nog niet bestaat
    performer_folder = save_path_base
    if not os.path.exists(performer_folder):
        os.makedirs(performer_folder)

    downloaded_paths = []

    for index, url in enumerate(urls):
        save_path = os.path.join(performer_folder, f"{performer_data['name']}_{index + 1}.jpg")
        if os.path.exists(save_path):
            print(f"Skipped downloading image, file already exists: {save_path}")
            downloaded_paths.append(save_path)
            continue

        try:
            print(f"Attempting to download image from {url}")
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, stream=True)

            if response.status_code == 200:
                with open(save_path, 'wb') as out_file:
                    for chunk in response.iter_content(1024):
                        out_file.write(chunk)

                with Image.open(save_path) as img:
                    img = img.convert("RGB")
                    img.save(save_path, "JPEG", quality=85, optimize=True)
                print(f"Downloaded and optimized image: {save_path}")
                downloaded_paths.append(save_path)
            else:
                print(f"Failed to download image, status code: {response.status_code}")

        except Exception as e:
            print(f"Error downloading image: {e}")
            downloaded_paths.append(None)

    return downloaded_paths

def write_to_json(performer_data):
    try:
        performer_name = performer_data.get('name', None)
        if not performer_name:
            print(f"Performer name is missing for {performer_data}. Skipping.")
            return

        print(f"Writing performer: {performer_name}")
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []

        # Voeg nieuwe data toe als performer nog niet bestaat
        if isinstance(existing_data, list):
            if performer_name not in [existing.get('name') for existing in existing_data]:
                existing_data.append(performer_data)
            else:
                for existing_performer in existing_data:
                    if existing_performer['name'] == performer_name:
                        existing_performer.update(performer_data)
                        break

        # Schrijf de data naar de JSON file
        with open(JSON_PATH, 'w', encoding='utf-8') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

        print(f"Data voor {performer_name} is succesvol weggeschreven.")

    except KeyError as e:
        print(f"KeyError: {e} - 'name' ontbreekt in performer data.")
    except Exception as e:
        print(f"Fout bij het wegschrijven naar JSON: {e}")

def main():
    max_performers = 1500000
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    get_pornhub_performers(max_performers)

if __name__ == '__main__':
    main()