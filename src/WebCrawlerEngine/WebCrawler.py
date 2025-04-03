import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from collections import deque
import json
import time

class WebCrawler:
    def __init__(self, start_url, max_depth=2, output_file="image_data.json"):
        self.start_url = start_url
        self.max_depth = max_depth
        self.output_file = output_file
        self.visited = set()
        self.visited_map = {}
        self.queue = deque([(start_url, 0)])
        self.image_data = []
        self.current_page = 0

    def fetch_page(self, url):
        """Fetch HTML content of a given URL."""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                print(f"Failed to fetch {url} (Status Code: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
        return None

    def parse_page(self, url, html, depth):
        """Extract images and metadata from div#bodyContent."""
        soup = BeautifulSoup(html, "html.parser")
        body_content = soup.find("div", id="bodyContent")
        body_text = body_content.get_text(strip=True) if body_content else None

        if not body_content:
            print(f"No body content found for {url}. Skipping...")
            return

        title = soup.find("h1", id="firstHeading").get_text(strip=True) if soup.find("h1", id="firstHeading") else "No title"
        categories = [cat.get_text(strip=True) for cat in soup.select("#mw-normal-catlinks ul li a")]

        images_on_page = []

        # Extract Infobox Image (Primary species image)
        infobox = soup.find("table", class_="infobox biota")
        if infobox:
            surrounding_text = infobox.find_next_sibling("p").get_text(strip=True) if infobox.find_next_sibling("p") else title
            for img_tag in infobox.find_all("img"):
                img_url = urljoin(url, img_tag["src"])
                alt_text = img_tag.get("alt", "").strip()
                images_on_page.append({
                    "page_url": url,
                    "page_title": title,
                    "categories": categories,
                    "image_url": img_url,
                    "alt_text": alt_text,
                    "body_text": body_text if body_text else "No body text",
                    "surrounding_text": surrounding_text,
                    "caption": "Infobox Image"
                })

        # Extract Thumbnail Images with Captions
        for thumb_div in body_content.find_all("div", class_="thumb"):
            for img_tag in thumb_div.find_all("div", class_="thumbimage"):
                for link in img_tag.find_all("a", href=True):
                    if not link.find("img"):
                        continue
                    href_value = link.find("img")['src']
                    img_url = urljoin(url, href_value)
                    alt_text = img_tag.get("alt", "").strip()
                    title_text = img_tag.get("title", "").strip()
                    caption_div = img_tag.find_next_sibling("div", class_="thumbcaption")
                    caption_text = caption_div.get_text(strip=True) if caption_div else "No caption available"
                    surrounding_text = thumb_div.find_next_sibling("p").get_text(strip=True) if thumb_div.find_next_sibling("p") else "No surrounding text"
                    images_on_page.append({
                        "page_url": url,
                        "page_title": title,
                        "categories": categories,
                        "image_url": img_url,
                        "alt_text": alt_text,
                        "title_text": title_text,
                        "body_text": body_text if body_text else "No body text",
                        "surrounding_text": surrounding_text,
                        "caption": caption_text
                    })

        for thumb_div in body_content.find_all("figure"):
            for link in thumb_div.find_all("a", href=True):
                for img_tag in link.find_all("img"):
                    href_value = img_tag['src']
                    img_url = urljoin(url, href_value)
                    alt_text = img_tag.get("alt", "") if img_tag else ""
                    title_text = img_tag.get("title", "") if img_tag else ""
                    caption_div = None
                    caption_div = link.find_next_sibling("div", class_="thumbcaption") or link.find_next_sibling("figcaption")
                    caption_text = caption_div.get_text(strip=True) if caption_div else "No caption available"
                    previous_p = thumb_div.find_previous_sibling("p").get_text()if thumb_div.find_previous_sibling("p") else ""
                    next_p = thumb_div.find_next_sibling("p").get_text() if thumb_div.find_next_sibling("p") else ""
                    surrounding_text = f"{previous_p} {next_p}" if previous_p or next_p else "No surrounding text"
                    images_on_page.append({
                        "page_url": url,
                        "page_title": title,
                        "categories": categories,
                        "image_url": img_url,
                        "alt_text": alt_text,
                        "title_text": title_text,
                        "body_text": body_text if body_text else "No body text",
                        "surrounding_text": surrounding_text,
                        "caption": caption_text
                    })

        self.image_data.extend(images_on_page)

        # Store visited page information
        self.visited_map[url] = {"depth": depth, "visit_count": self.visited_map.get(url, {}).get("visit_count", 0) + 1}

        # Save after processing each page
        self.save_to_json()

    def crawl(self):
        """Perform BFS crawling on Wikipedia."""
        while self.queue:
            if self.current_page > 1000: 
                break
            url, depth = self.queue.popleft()
            if url in self.visited or depth > self.max_depth:
                continue

            self.current_page += 1
            print(f"{self.current_page} Crawling: {url} (Depth: {depth})")
            self.visited.add(url)

            html = self.fetch_page(url)
            if html:
                self.parse_page(url, html, depth)

                # Find new Wikipedia links only inside bodyContent
                soup = BeautifulSoup(html, "html.parser")
                body_content = soup.find("div", id="bodyContent")
                if body_content:
                    for link in body_content.find_all("a", href=True):
                        full_url = urljoin(url, link["href"])
                        if full_url.startswith("https://en.wikipedia.org/wiki/") and ":" not in link["href"] and "#" not in link["href"]:
                            self.queue.append((full_url, depth + 1))

            time.sleep(1)  # Avoid aggressive requests

    def save_to_json(self):
        """Save extracted image metadata and visited pages to a JSON file."""
        data = {
            "image_data": self.image_data,
            "visited_map": self.visited_map
        }

        try:
            # Read existing data
            with open("image_data.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        # Append new data
        existing_data.update(data)

        # Save back as a JSON array
        with open("image_data.json", "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)


        print("Data appended successfully!")

    def get_results(self):
        """Return the extracted image metadata."""
        return self.image_data

    def get_visited_map(self):
        """Return the dictionary of visited URLs with metadata."""
        return self.visited_map


# Example Usage
start_url = "https://en.wikipedia.org/wiki/Category:Domestic_implements"
crawler = WebCrawler(start_url, max_depth=3)
crawler.crawl()

# Print extracted image metadata
print("\nExtracted Image Metadata:")
for img in crawler.get_results():
    print(img)

# Print visited URL map
print("\nVisited URLs:")
for url, info in crawler.get_visited_map().items():
    print(f"URL: {url}, Depth: {info['depth']}, Visits: {info['visit_count']}")