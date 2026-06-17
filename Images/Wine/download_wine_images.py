import requests
import os
import time

# Your Unsplash API credentials
# Get your key at: https://unsplash.com/developers
ACCESS_KEY = "mtHRCpMpCyrNT8-cT_e93x-ragpq8nk3qQR7wKMBNbk"

# Create directory for images
os.makedirs("wine_bottles", exist_ok=True)

# Unsplash API setup
base_url = "https://api.unsplash.com/search/photos"
headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}

downloaded = 0
page = 1
per_page = 30  # Max per request

while downloaded < 300:
    # Search for wine bottles
    params = {
        "query": "wine bottle",
        "page": page,
        "per_page": per_page,
        "orientation": "portrait"  # optional: portrait, landscape, squarish
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    data = response.json()
    
    if not data.get("results"):
        print("No more results")
        break
    
    for photo in data["results"]:
        if downloaded >= 300:
            break
            
        # Get the regular size URL (you can use 'full', 'regular', 'small', 'thumb')
        img_url = photo["urls"]["regular"]
        
        # Download the image
        img_response = requests.get(img_url)
        filename = f"wine_bottles/wine_bottle_{downloaded+1}.jpg"
        
        with open(filename, "wb") as f:
            f.write(img_response.content)
        
        downloaded += 1
        print(f"Downloaded {downloaded}/300: {filename}")
        
        # Rate limiting - be nice to the API
        time.sleep(0.5)
    
    page += 1
    time.sleep(1)  # Pause between pages

print(f"\nDone! Downloaded {downloaded} images.")
