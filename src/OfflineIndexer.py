import json
import pickle
import time
import cv2
import requests
import numpy as np
from threading import Lock
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from ultralytics import YOLO  
from Document import Document
from BestMatching25 import BM25
from TF_IDF_Builder import TF_IDF_Builder
from TextPreprocessor import TextPreprocessor

# Load YOLO model for object detection
model = YOLO("yolo12n.pt") 

class OfflineIndexer:
    def __init__(self):
        self.documents = []
        self.tf_idf = TF_IDF_Builder(TextPreprocessor())
        self.object_metadata = {}  # Store detected objects separately
        self.model_lock = Lock()  # Lock for YOLO model to ensure thread safety
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fetch_image(self, image_url):
        """Fetch and process an image from a URL, ensuring correct format for YOLO."""
        try:
            headers = {'User-Agent': 'ImageSearch101/0.0.1'}
            response = requests.get(image_url, stream=True, headers=headers)
            response.raise_for_status()  # Raise error for failed requests

            image_array = np.frombuffer(response.content, np.uint8)  
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  

            if image is None:
                print(f"Failed to decode image: {image_url}")
                return None

            return image
        except Exception as e:
            print(f"Error fetching image {image_url}: {e}")
            return None

    def get_dominant_colors(self, image, top_n=3):
        """Get the top N dominant colors of an image using numpy."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        pixels = image.reshape(-1, 3)
        unique, counts = np.unique(pixels, axis=0, return_counts=True)
        sorted_indices = np.argsort(-counts)  # Sort by frequency in descending order
        dominant_colors = [tuple(unique[i]) for i in sorted_indices[:top_n]]
        return dominant_colors

    def detect_objects_and_metadata(self, image, doc):
        """Detect objects, image size, and dominant color in an image."""
        try:
            image_url = doc.path  # Use the image URL from the document
            # Get image size (width, height)
            height, width, _ = image.shape
            image_size = {"width": width, "height": height}

            # Get dominant color using numpy
            dominant_colors = self.get_dominant_colors(image)

            # Run YOLO detection
            with self.model_lock:  # Ensure YOLO model is accessed by one thread at a time
                results = model(image)  
            detected_objects = set()

            for r in results:
                for c in r.boxes.cls:
                    detected_objects.add(model.names[int(c)])  # Convert class index to label

            metadata = {
                "image_url": image_url,
                "image_size": [int(value) for value in image_size.values()],  # Convert to Python int for JSON serialization
                "dominant_colors": [list(map(int, color)) for color in dominant_colors],  # Ensure values are JSON serializable
                "detected_objects": list(detected_objects),
                "categories": doc.categories,
                "caption": doc.caption,
                "alt_text": doc.alt_text,
                "page_title": doc.file_name
            }

            return metadata
        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            return None

    def preprocess_bm25(self, text):
        """Tokenizes, removes stopwords, and lemmatizes the text for BM25 indexing."""
        tokens = word_tokenize(text.lower())  # Tokenization
        tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
        tokens = [word for word in tokens if word not in self.stop_words]  # Remove stopwords
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
        return tokens

    def build_index(self):
        """Reads JSON data, processes text, performs object detection, and builds a BM25 index."""
        with open("image_data.json", "r", encoding="utf-8") as f:
            try:
                file = json.load(f)
            except json.JSONDecodeError:
                print("Error loading JSON file.")
                return

        image_url_to_text = {}

        for inner_item in file.get("image_data", []):
            page_url = inner_item.get("page_url", "")
            page_title = inner_item.get("page_title", "")
            categories = inner_item.get("categories", "")
            image_url = inner_item.get("image_url", "")
            alt_text = inner_item.get("alt_text", "")
            title_text = inner_item.get("title_text", "")
            caption = inner_item.get("caption", "")
            body_text = inner_item.get("body_text", "")
            surrounding_text = inner_item.get("surrounding_text", "")

            combined_text = f"{page_title} {categories} {alt_text} {title_text} {surrounding_text} {caption}"

            doc = Document(page_url, page_title, image_url, combined_text, combined_text, ".html", 
                           categories,caption, alt_text)
            self.documents.append(doc)

        print(f"Total documents indexed: {len(self.documents)}")

        for doc in self.documents[:1100]: 
            if doc.path:  
                if doc.path and doc.path.startswith("http"):  # Ensure doc.path is a valid URL
                    image = self.fetch_image(doc.path)  # Fetch image from URL
                else:
                    print(f"Invalid or missing URL for document: {doc}")
                    continue
                time.sleep(1)  # Avoid aggressive requests
                if image is not None:
                    metadata = self.detect_objects_and_metadata(image, doc)  # Get metadata
                    self.object_metadata[doc.path] = metadata  # Store metadata
                    if metadata:
                        # Add detected objects to weighted text
                        detected_objects_str = " ".join(metadata["detected_objects"])
                        doc.original_text += " " + detected_objects_str + " "
                    
                    doc.preprocessed_text = self.preprocess_bm25(doc.original_text)
                    image_url_to_text[doc.path] = doc.preprocessed_text

        corpus = list(image_url_to_text.values())
        if not corpus:
            print("Warning: No text was indexed!")
            return

        bm25 = BM25Okapi(corpus)

        # ✅ Save detected objects, size, and color metadata
        with open("detected_objects_metadata.json", "w") as f:
            json.dump(self.object_metadata, f, indent=4)

        print(f"Metadata stored successfully for {len(self.object_metadata)} images.")

        # ✅ Save BM25 index
        with open("bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)

        with open("image_url_to_text.pkl", "wb") as f:
            pickle.dump(image_url_to_text, f)

        print(f"BM25 index stored successfully with {len(corpus)} images.")
# Main Program
if __name__ == "__main__":
    indexer = OfflineIndexer()
    indexer.build_index()