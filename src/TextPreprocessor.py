import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

nltk.download('punkt')
nltk.download('stopwords')
nltk.download("wordnet")

# Load spaCy model for semantic similarity
nlp = spacy.load("en_core_web_sm")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.max_synonyms = 3

    def get_wordnet_synonyms(self, word):
        """Fetch synonyms from WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))  # Replace underscores
        return list(synonyms)[: self.max_synonyms]  # Limit synonyms

    def get_spacy_synonyms(self, word):
        """Find semantically similar words using spaCy word vectors."""
        word_token = nlp(word)
        similar_words = []
        for candidate in nlp.vocab:
            if candidate.is_alpha and candidate.has_vector:  # Filter valid words
                similarity = word_token.similarity(nlp(candidate.text))
                if similarity > 0.6:  # Only include words with strong similarity
                    similar_words.append((candidate.text, similarity))
        similar_words.sort(key=lambda x: x[1], reverse=True)
        return [word[0] for word in similar_words[: self.max_synonyms]]

    def synonym_expansion(self, word):
        """Expand words with synonyms using WordNet and spaCy."""
        synonyms = set()
        synonyms.update(self.get_wordnet_synonyms(word))
        synonyms.update(self.get_spacy_synonyms(word))

        if synonyms:
            return f"({word} {' '.join(synonyms)})"
        return word
    
    def extract_named_entities(self, text):
        """Extract named entities from text using spaCy's NER."""
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities
    
    def preprocess(self, text,isQuery=False):
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove punctuation and stopwords
        tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        # Stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        # Lemmatization
        #tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        #Expand for query
        if isQuery:
            tokens = [self.synonym_expansion(word) for word in tokens]  # Synonyms
        
        # Named Entity Recognition
        tokens.extend(self.extract_named_entities(text))

        return " ".join(tokens)