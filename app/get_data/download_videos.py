import random
import os
import yt_dlp  # Using yt-dlp to download videos
from pornhub_api import PornhubApi

# Initialize PornhubAPI for searching
api = PornhubApi()

# Fetch 5 random tags (you can customize this)
tags = random.sample(api.video.tags("f").tags, 5)

# Pick a random category
category = random.choice(api.video.categories().categories)

# Search for videos with random tags and category
result = api.search.search(ordering="mostviewed", tags=tags, category=category)

# Path to the input_videos folder
output_folder = './app/input_videos/'

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through the search results
for vid in result.videos:  # Accessing the videos attribute correctly
    # Print the video details to understand the structure

    # Assuming vid is a dictionary, access its attributes
    video_id = vid.video_id  # Corrected attribute access
    duration = vid.duration
    title = vid.title
    url = vid.url

    # Print the video_data to understand its structure
    print("Video data:")
    print(f"ID: {video_id}, Duration: {duration}, Title: {title}, URL: {url}")

    # Convert duration to seconds if necessary
    duration_seconds = int(duration.split(':')[0]) * 60 + int(duration.split(':')[1])

    # Check if the duration is between 10 minutes (600 seconds) and 30 minutes (1800 seconds)
    if 600 <= duration_seconds <= 1800:
        print(f"Found video: {title} ({duration_seconds // 60} minutes)")
        print(f"URL: {url}")

        # Define the path for saving the video
        video_path = os.path.join(output_folder, f"{title}.mp4")

        try:
            # Use yt-dlp to download the video
            ydl_opts = {
                'outtmpl': video_path,
                'format': 'bestvideo+bestaudio/best',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            print(f"Video '{title}' downloaded successfully at {video_path}!")

        except Exception as e:
            print(f"Failed to download video '{title}': {e}")