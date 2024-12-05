import cloudscraper
import requests
from bs4 import BeautifulSoup
import json
import os
import re
from PIL import Image
import concurrent.futures
import time

def get_pornhub_performers(max_performers=15000):
    performers = []
    base_url = "https://nl.pornhub.com/pornstars?performerType=pornstar&page="
    scraper = cloudscraper.create_scraper()

    # Voeg een User-Agent header toe voor meer authenticiteit
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

        # Controleer of de pagina succesvol is opgehaald
        soup = BeautifulSoup(response.content, 'html.parser')
        performer_cards = soup.find_all('li', class_='pornstarLi performerCard')

        # Stop als er geen performerkaarten meer worden gevonden
        if not performer_cards:
            # Controleer of de HTML wijst op een lege pagina
            html_snippet = soup.prettify()[:200]
            if "No performer cards found" in html_snippet or "<!DOCTYPE html>" in html_snippet:
                print("No performer cards found. Here is a snippet of the HTML: ")
                print(html_snippet)
                break

        page_data = []  # Opslaan van performers per pagina

        for card in performer_cards:
            name_tag = card.find('span', class_='pornStarName performerCardName')
            img_tag = card.find('img', attrs={'data-thumb_url': True})
            verified_tag = card.find('span', class_='verifiedPornstar')

            # Controleer of de afbeelding geen standaardafbeelding is
            if img_tag and "default" in img_tag['data-thumb_url']:
                continue

            # Voeg alleen geverifieerde performers met geldige afbeelding toe
            if name_tag and img_tag and verified_tag:
                name = name_tag.get_text(strip=True)
                formatted_name = name.replace(' ', '_')  # Format de naam als name_achternaam
                img_url = img_tag['data-thumb_url']
                performer_data = {"name": formatted_name, "img_url": img_url}
                performers.append(performer_data)
                page_data.append(performer_data)  # Voeg toe aan page_data
                print(f"Added performer: {formatted_name}")

            # Stop als het maximumaantal performers is bereikt
            if len(performers) >= max_performers:
                break

        if page_data:
            # Update de JSON na elke pagina
            try:
                # Laad bestaande data als het bestand al bestaat
                if os.path.exists('performers_data.json'):
                    with open('performers_data.json', 'r', encoding='utf-8') as json_file:
                        existing_data = json.load(json_file)
                else:
                    existing_data = []

                # Voeg de nieuwe data van de pagina toe aan de bestaande data
                existing_data.extend(page_data)

                # Sla de bijgewerkte gegevens op in het JSON-bestand
                with open('performers_data.json', 'w', encoding='utf-8') as json_file:
                    json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
                print(f"Updated data for page {page}")
            except IOError as e:
                print(f"Error saving JSON file for page {page}: {e}")

        page += 1
        time.sleep(2)  # Pauze om rate-limiting te voorkomen

    return performers

def get_iafd_details(performer_name):
    formatted_name = performer_name.replace(' ', '+')
    scraper = cloudscraper.create_scraper()
    search_url = f"https://www.iafd.com/results.asp?searchtype=comprehensive&searchstring={formatted_name}"
    try:
        response = scraper.get(search_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    soup = BeautifulSoup(response.content, 'html.parser')

    for link in soup.find_all('a', href=True):
        if '/person.rme/id=' in link['href']:
            performer_url = 'https://www.iafd.com' + link['href']
            try:
                performer_response = scraper.get(performer_url)
                performer_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                continue
            performer_soup = BeautifulSoup(performer_response.content, 'html.parser')

            details = {}
            headshot_div = performer_soup.find('div', id='headshot')
            if headshot_div:
                img_tag = headshot_div.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    details['image_url'] = img_tag['src']

            # Verzamel biografische gegevens uit de "vitalbox"
            vitalbox = performer_soup.find('div', id='vitalbox')
            if vitalbox:
                bioheadings = vitalbox.find_all('p', class_='bioheading')
                biodata = vitalbox.find_all('p', class_='biodata')
                for heading, data in zip(bioheadings, biodata):
                    details[heading.text.strip()] = data.text.strip()

            return details

    return None

def download_and_optimize_image(url, save_path):
    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as out_file:
                for chunk in response.iter_content(1024):
                    out_file.write(chunk)

            # Converteer en optimaliseer de afbeelding naar JPEG
            with Image.open(save_path) as img:
                img = img.convert("RGB")
                img.save(save_path, "JPEG", quality=85, optimize=True)
        else:
            print(f"Failed to download image from {url}")
    except Exception as e:
        print(f"Error downloading image: {e}")

def main():
    max_performers = 15000
    performers = get_pornhub_performers(max_performers)
    output_dir = "app/datasets/pornstar_images"
    json_path = "app/datasets/performers_data.json"

    # Controleer en maak de outputmap indien nodig
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []

    # Gebruik een processpool om afbeeldingen te downloaden en gegevens te verwerken
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for performer in performers:
            name = performer['name']
            img_url = performer['img_url']

            save_path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            futures[executor.submit(download_and_optimize_image, img_url, save_path)] = name

            # Haal extra details op van IAFD
            iafd_details = get_iafd_details(name)
            if iafd_details:
                iafd_details['name'] = name
                iafd_details['pornhub_img_url'] = img_url
                data.append(iafd_details)
            else:
                print(f"Performer {name} not found on IAFD. Skipping.")

        # Verwerk de resultaten van de threads
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"Downloaded and optimized image for {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

    # Sla de verzamelde gegevens op in een JSON-bestand
    try:
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error saving JSON file: {e}")

    print(f"Data collection complete. JSON saved at {json_path}")

if __name__ == "__main__":
    main()