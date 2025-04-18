import json
from flask import Flask, request, render_template
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests
import numpy as np
import time
from nltk.corpus import stopwords, wordnet
import spacy

nltk.download("punkt")  # Ensure tokenizer is available
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Load BM25 index & corpus on startup
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("image_url_to_text.pkl", "rb") as f:
    image_url_to_text = pickle.load(f)

# ✅ Load Object Detection Metadata (detected objects per image)
with open("detected_objects_metadata.json", "r") as f:
    metadata = json.load(f)  # {image_url: [list of detected objects]}

RESULTS_PER_PAGE = 20  # Number of results per page
SCORE_THRESHOLD = 0.25  # Minimum normalized BM25 score for inclusion
BOOST_FACTOR = 3  # Boost multiplier for images with detected objects
MAX_SYNONYMS = 3  # Maximum number of synonyms to fetch
def get_wordnet_synonyms(word):
    """Fetch synonyms from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Replace underscores
    return list(synonyms)[: MAX_SYNONYMS]  # Limit synonyms

def get_spacy_synonyms(word):
    """Find semantically similar words using spaCy word vectors."""
    word_token = str(word)
    similar_words = []
    for candidate in nlp.vocab:
        if candidate.is_alpha and candidate.has_vector:  # Filter valid words
            similarity = word_token.similarity(nlp(candidate.text))
            if similarity > 0.6:  # Only include words with strong similarity
                similar_words.append((candidate.text, similarity))
    similar_words.sort(key=lambda x: x[1], reverse=True)
    return [word[0] for word in similar_words[: MAX_SYNONYMS]]

def synonym_expansion(word):
    """Expand words with synonyms using WordNet and spaCy."""
    synonyms = set()
    synonyms.update(get_wordnet_synonyms(word))
    synonyms.update(get_spacy_synonyms(word))

    if synonyms:
        return f"({word} {' '.join(synonyms)})"
    return word
def is_color_match(image_color, selected_color):
    # Convert colors to RGB
    def hex_to_rgb(hex_color):
        return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    selected_rgb = hex_to_rgb(selected_color)

    # Check if all RGB values are within 50 of each other
    threshold = 100
    return all(abs(a - b) <= threshold for a, b in zip(image_color, selected_rgb))

@app.route("/")
def home():
    return render_template("index.html")  # Search page

def get_color_name(list_of_colors):
    # Convert rgb to hex
    hex_colors= []
    for rgb in list_of_colors:
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        hex_colors.append(hex_color)
    return hex_colors

@app.route("/search_results", methods=["GET"])
def search_results():
    query = request.args.get("query", "").strip().lower()
    page = int(request.args.get("page", 1))  # Get page number, default is 1
    size_filter = request.args.get("size")  # Example: "medium"
    color_filter = request.args.get("color")  # Example: "#FF5733"
    category_filter = request.args.get("category")  # Example: "animals"

    if not query:
        return render_template("results.html", results=[], query=query, page=page, total_pages=0, total_results=0)

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(query.lower())
    tokens = [word for word in tokens if word.isalnum()]  # Remove non-alphanumeric characters
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    tokens.extend([synonym_expansion(word) for word in tokens])  # Synonym Expansion - Can improve query recall
    print(tokens)
    scores = bm25.get_scores(tokens)  # BM25 similarity scores

    # ✅ Normalize BM25 scores
    min_score, max_score = min(scores), max(scores)
    normalized_scores = [(s - min_score) / (max_score - min_score + 1e-9) for s in scores]  # Avoid division by zero
    results = []

    if all(score == 0 for score in scores):
        return render_template("results.html", results=[], query=query, page=page, total_pages=0, total_results=0)

    for i, (image_url, text) in enumerate(image_url_to_text.items()):
        score = scores[i]
        norm_score = normalized_scores[i]  # Use normalized score
        # ✅ Apply BM25 cut-off threshold
        if norm_score >= SCORE_THRESHOLD:
            final_score = norm_score  # Base score
            
            # ✅ Boost score if detected objects match query
            metadata_for_image = metadata.get(image_url, {})
            metadata_list = metadata_for_image.get("detected_objects", [])  # Access the "objects" key
            # Check how many tokens match detected objects and apply proportional boost
            matching_tokens = sum(1 for token in tokens if token in metadata_list)
            if matching_tokens > 0:
                 final_score *= (BOOST_FACTOR * matching_tokens)  # Apply boost proportional to matches

            # ✅ Add metadata to results
            results.append({
                "image_url": image_url,  
                "description": text,
                "image_size": metadata_for_image.get("image_size", [0, 0]),  # Use default size if not available
                "dominant_colors": get_color_name(metadata_for_image.get("dominant_colors", [])),  # Use default color if not available
                "color_rgb": metadata_for_image.get("dominant_colors", [0, 0, 0]),  # Use default color if not available
                "score": final_score,  # Use boosted score if applicable
                "detected_objects": metadata_for_image.get("detected_objects", []),  # Use default empty list if not available
                "categories": metadata_for_image.get("categories", []),  # Use default empty list if not available
                "caption": metadata_for_image.get("caption", ""),  # Use default empty string if not available
                "alt_text": metadata_for_image.get("alt_text", ""),  # Use default empty string if not available
                "page_title": metadata_for_image.get("page_title", "")  # Use default empty string if not available
            })

    # ✅ Filter results based on size and color
    filtered_results = []
    for result in results:
        image_url = result["image_url"]
        
        size = result["image_size"]

        # Filter images based on the selected size
        # Define size thresholds for small, medium, and large
        x=1000 
        y = 1000
        if size_filter == "small":
            min_size, max_size = 0, 100
        elif size_filter == "medium":
            min_size, max_size = 101, 500
        elif size_filter == "large":
            min_size, max_size = 501, 2000
        else:
            min_size, max_size = 0, float('inf')  # Default range for no filter

        # Skip images that don't fall within the size range
        if not (min_size <= size[0] <= max_size and min_size <= size[1] <= max_size):
            continue

        # Filter images based on the selected color
        # Skip images that don't match the selected color
        colors = result["color_rgb"]
        if color_filter and not any(is_color_match(color, color_filter) for color in colors):
            continue

        # Filter images based on the selected categories
        if category_filter and category_filter not in result["categories"]:
            continue
            
        # Append the result if it passes all filters
        filtered_results.append(result)

    # ✅ Sort results by final score (highest first)
    filtered_results.sort(key=lambda x: x["score"], reverse=True)
    total_results = len(filtered_results)
    total_pages = (total_results // RESULTS_PER_PAGE) + (1 if total_results % RESULTS_PER_PAGE else 0)

    # Paginate results
    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    paginated_results = filtered_results[start:end]

    # Extract unique categories
    available_categories = set()
    for result in results:
        for category in result["categories"]:
            available_categories.add(category)

    # Pass the available_categories to the template
    return render_template(
        "results.html",
        results=paginated_results,
        query=query,
        page=page,
        total_pages=total_pages,
        total_results=total_results,
        available_categories=list(available_categories)  # Add the categories to the context
    )

if __name__ == "__main__":
    app.run(debug=True)