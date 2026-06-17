import requests
import os
import time

# Your Unsplash API credentials
# Get your key at: https://unsplash.com/developers
ACCESS_KEY = "mtHRCpMpCyrNT8-cT_e93x-ragpq8nk3qQR7wKMBNbk"

# Create directory for images
os.makedirs("Non-Wine", exist_ok=True)

# Unsplash API setup
base_url = "https://api.unsplash.com/search/photos"
headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}

# Search terms for non-wine bottles (for classification dataset)
search_terms = ["beer bottle", "whiskey bottle", "vodka bottle", "perfume bottle", "olive oil bottle", "juice bottle", "soda bottle", "water bottle glass", "ketchup bottle", "sauce bottle", "vinegar bottle", "oil bottle"]

downloaded = 0
target = 300

for search_term in search_terms:
    if downloaded >= target:
        break
    
    page = 1
    per_page = 30
    
    print(f"\nSearching for: {search_term}")
    
    while downloaded < target:
        params = {
            "query": search_term,
            "page": page,
            "per_page": per_page,
            "orientation": "portrait"
        }
        
        response = requests.get(base_url, headers=headers, params=params)
        data = response.json()
        
        if not data.get("results"):
            break
        
        for photo in data["results"]:
            if downloaded >= target:
                break
            
            img_url = photo["urls"]["regular"]
            img_response = requests.get(img_url)
            filename = f"Non-Wine/non_wine_{downloaded+1}.jpg"
            
            with open(filename, "wb") as f:
                f.write(img_response.content)
            
            downloaded += 1
            print(f"Downloaded {downloaded}/{target}: {filename}")
            time.sleep(0.5)
        
        page += 1
        time.sleep(1)

print(f"\nDone! Downloaded {downloaded} images to Non-Wine folder.")
