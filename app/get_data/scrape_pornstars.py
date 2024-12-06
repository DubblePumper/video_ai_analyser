import cloudscraper
import requests
from bs4 import BeautifulSoup
import json
import os
import time
from PIL import Image

# Global variables for paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Dit is de map 'app'

# De paden voor datasets en afbeeldingen worden relatief vanaf de 'app' directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets', 'pornstar_images')
JSON_PATH = os.path.join(BASE_DIR, 'datasets', 'performers_data.json')

def get_pornhub_performers(max_performers=15000):
    performers = []
    base_url = "https://nl.pornhub.com/pornstars?performerType=pornstar&page="
    scraper = cloudscraper.create_scraper()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36"
    }

    page = 1
    while len(performers) < max_performers:
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
            img_tag = card.find('img', attrs={'data-thumb_url': True})
            verified_tag = card.find('span', class_='verifiedPornstar')

            if img_tag and "default" in img_tag['data-thumb_url']:
                continue

            if name_tag and img_tag and verified_tag:
                # Extract the name text and remove extra nested tags
                name = name_tag.get_text()
                # Replace spaces with underscores and handle multiple parts of the name
                formatted_name = "_".join(name.split())
                
                img_url = img_tag['data-thumb_url']
                performer_data = {"name": formatted_name, "img_url": img_url}
                
                # Skip if performer already exists in the list based on name or image URL
                if any(existing['name'] == formatted_name or existing['img_url'] == img_url for existing in performers):
                    print(f"Skipped duplicate performer (name or image): {formatted_name}")
                    continue

                performers.append(performer_data)
                page_data.append(performer_data)
                print(f"page: {page} | Added performer: {formatted_name}")

                # Download and optimize the image immediately
                save_path = os.path.join(OUTPUT_DIR, f"{formatted_name}.jpg")
                download_and_optimize_image(img_url, save_path)

        if page_data:
            try:
                # Save the page data immediately to the JSON file
                if os.path.exists(JSON_PATH):
                    with open(JSON_PATH, 'r', encoding='utf-8') as json_file:
                        existing_data = json.load(json_file)
                else:
                    existing_data = []

                # Filter out any duplicates already in the existing data
                for new_data in page_data:
                    if not any(existing['name'] == new_data['name'] or existing['img_url'] == new_data['img_url'] for existing in existing_data):
                        existing_data.append(new_data)

                with open(JSON_PATH, 'w', encoding='utf-8') as json_file:
                    json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
                print(f"Updated data for page {page}")
                print(f"Total performers: {len(performers)}")
                print(f"------------------------------------------------------------------------------")
            except IOError as e:
                print(f"Error saving JSON file for page {page}: {e}")

        if len(performers) >= max_performers:
            break

        page += 1
        time.sleep(2)

    return performers

def download_and_optimize_image(url, save_path):
    # Skip downloading if the image already exists
    if os.path.exists(save_path):
        print(f"Skipped downloading image, file already exists")
        return

    try:
        print(f"Attempting to download image from {url}")  # Debug: print when downloading starts
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, stream=True)

        # Check if the response status code is OK (200)
        if response.status_code == 200:
            with open(save_path, 'wb') as out_file:
                for chunk in response.iter_content(1024):
                    out_file.write(chunk)

            # Now, optimize the image
            with Image.open(save_path) as img:
                img = img.convert("RGB")
                img.save(save_path, "JPEG", quality=85, optimize=True)
            print(f"Downloaded and optimized image")
        else:
            print(f"Failed to download image, status code: {response.status_code}")
            print(f"Response content: {response.text[:200]}")  # Print part of the response for debugging
    except Exception as e:
        print(f"Error downloading image: {e}")

def main():
    max_performers = 15000
    performers = get_pornhub_performers(max_performers)

    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    data = []

    for performer in performers:
        name = performer['name']
        img_url = performer['img_url']
        print(f"Processing performer: {name} with image")  # Debug: print performer info

        save_path = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}.jpg")
        download_and_optimize_image(img_url, save_path)

        # Skip if performer already exists in data
        if not any(existing['name'] == name or existing['img_url'] == img_url for existing in data):
            data.append({"name": name, "img_url": img_url})
        else:
            print(f"Skipped duplicate performer (name or image) in final data: {name}")

    # Writing all data to the JSON file at the end
    with open(JSON_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()