import cloudscraper
from bs4 import BeautifulSoup
import json
import os
import re
from PIL import Image
import concurrent.futures

def get_pornhub_performers(max_performers=15000):
    # Functie om een lijst van performers van Pornhub te verkrijgen.
    performers = []
    base_url = "https://nl.pornhub.com/pornstars?performerType=pornstar&page="
    scraper = cloudscraper.create_scraper()

    page = 1
    while len(performers) < max_performers:
        url = f"{base_url}{page}"
        print(f"Scraping page {page}: {url}")
        response = scraper.get(url)

        # Controleer of de pagina succesvol is opgehaald
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}, status code: {response.status_code}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        performer_cards = soup.find_all('li', class_='pornstarLi performerCard')  # Correcte klasse gebruikt

        # Stop als er geen performerkaarten meer worden gevonden
        if not performer_cards:
            print("No more performer cards found.")
            break

        for card in performer_cards:
            name_tag = card.find('span', class_='pornStarName performerCardName')  # Correcte selector
            img_tag = card.find('img')

            # Voeg de naam en afbeelding-URL toe als deze aanwezig zijn
            if name_tag and img_tag:
                name = name_tag.text.strip()
                img_url = img_tag.get('data-thumb_url') or img_tag.get('src')
                if 'verifiedPornstar' in str(card):
                    performers.append({"name": name, "img_url": img_url})

            # Stop als het maximumaantal performers is bereikt
            if len(performers) >= max_performers:
                break

        page += 1

    return performers

def get_iafd_details(performer_name):
    # Functie om details van een performer van IAFD te verkrijgen.
    formatted_name = performer_name.replace(' ', '+')
    scraper = cloudscraper.create_scraper()
    search_url = f"https://www.iafd.com/results.asp?searchtype=comprehensive&searchstring={formatted_name}"
    response = scraper.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    for link in soup.find_all('a', href=True):
        if '/person.rme/id=' in link['href']:
            performer_url = 'https://www.iafd.com' + link['href']
            performer_response = scraper.get(performer_url)
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
    # Functie om een afbeelding te downloaden en te optimaliseren.
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
    # Hoofdprogramma om performers te verzamelen en hun gegevens op te slaan.
    max_performers = 15000
    performers = get_pornhub_performers(max_performers)
    output_dir = "app/output_images"
    json_path = "app/performers_data.json"

    # Controleer en maak de outputmap indien nodig
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []

    # Gebruik een threadpool om afbeeldingen te downloaden en gegevens te verwerken
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

        # Verwerk de resultaten van de threads
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"Downloaded and optimized image for {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

    # Sla de verzamelde gegevens op in een JSON-bestand
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print(f"Data collection complete. JSON saved at {json_path}")

if __name__ == "__main__":
    main()