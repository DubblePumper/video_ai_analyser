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

def get_iafd_details(performer_name):
    iafd_base_url = "https://www.iafd.com/results.asp?searchtype=comprehensive&searchstring="
    search_url = iafd_base_url + performer_name.replace("_", " ").replace(" ", "+")
    print(f"Searching IAFD for performer: {performer_name.replace('_', ' ')}")

    try:           
        scraper = cloudscraper.create_scraper()
        response = scraper.get(search_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        performer_details = {}

        # Check if no results are found
        no_results = soup.find(string=lambda text: "No performers matched your search query." in text if text else False)
        if no_results:
            print(f"Performer not found on IAFD: {performer_name}")
            return None

        # Find the table that contains performer details
        table = soup.find('table', id='tblMal', class_='table display table-responsive')
        if not table:
            print(f"No performer table found: {performer_name}")
            return None

        # Find the row with the performer link
        row = table.find('a', href=lambda href: href and "/person.rme/" in href)
        if row:
            row = row.find_parent('td')  # Get the parent <td> that contains performer info
        
        if not row:
            print(f"No performer row found: {performer_name}")
            return None

        # Extract performer details from the row
        link = row.find('a')
        if link and 'href' in link.attrs:
            performer_details['iafd_profile_url'] = f"https://www.iafd.com{link['href']}"

        # Extract other details like name, aka, start year, etc.
        cells = row.find_all('td')
        
        if len(cells) > 0:
            name_cell = cells[0]
            if name_cell:
                performer_details['iafd_name'] = name_cell.text.strip()

        if len(cells) > 1:
            aka_cell = cells[1]
            if aka_cell:
                performer_details['iafd_aka'] = aka_cell.text.strip()

        if len(cells) > 2:
            start_cell = cells[2]
            if start_cell:
                performer_details['start_year'] = start_cell.text.strip()

        if len(cells) > 3:
            end_cell = cells[3]
            if end_cell:
                performer_details['end_year'] = end_cell.text.strip()

        if len(cells) > 4:
            titles_cell = cells[4]
            if titles_cell:
                performer_details['titles'] = titles_cell.text.strip()

        print(f"Retrieved IAFD details for {performer_name}: {performer_details}")
        return performer_details

    except Exception as e:
        print(f"Error retrieving IAFD details for {performer_name}: {e}")
        return None

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
                name = name_tag.get_text()
                formatted_name = "_".join(name.split())
                img_url = img_tag['data-thumb_url']

                performer_data = {"name": formatted_name, "img_url": img_url}

                # Haal IAFD-gegevens op en voeg toe als beschikbaar
                iafd_details = get_iafd_details(formatted_name)
                if iafd_details:
                    performer_data.update(iafd_details)

                if any(existing['name'] == formatted_name or existing['img_url'] == img_url for existing in performers):
                    print(f"Skipped duplicate performer (name or image): {formatted_name}")
                    continue

                performers.append(performer_data)
                page_data.append(performer_data)
                print(f"page: {page} | Added performer: {formatted_name}")

                save_path = os.path.join(OUTPUT_DIR, f"{formatted_name}.jpg")
                download_and_optimize_image(img_url, save_path)

        if page_data:
            try:
                if os.path.exists(JSON_PATH):
                    with open(JSON_PATH, 'r', encoding='utf-8') as json_file:
                        existing_data = json.load(json_file)
                else:
                    existing_data = []

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
    if os.path.exists(save_path):
        print(f"Skipped downloading image, file already exists")
        return

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
            print(f"Downloaded and optimized image")
        else:
            print(f"Failed to download image, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {e}")

def main():
    max_performers = 15000
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    get_pornhub_performers(max_performers)

if __name__ == '__main__':
    main()
