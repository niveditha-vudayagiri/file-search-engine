import math
from collections import Counter
from nltk.tokenize import word_tokenize
from TextPreprocessor import TextPreprocessor
import copy

class MultinomialLanguageModel:
    def __init__(self, tfidf_builder, trec, mu=2000, lambda_unk=0.0001):
        """
        Initialize the Language Model for Information Retrieval.
        :param tfidf_builder: An instance of the TF_IDF_Builder class.
        :param mu: Dirichlet smoothing parameter.
        :param lambda_unk: Probability mass for unknown words.
        """
        self.preprocessor = tfidf_builder.preprocessor
        self.tfidf_builder = tfidf_builder

        self.trec = trec
        self.mu = mu
        self.lambda_unk = lambda_unk  # Smoothing for unknown words
        self.documents = []

        self.total_terms = 0
        self.term_frequencies = Counter()
        self.doc_lengths = []
        self.doc_frequencies = {}  # Changed to a dictionary of dictionaries
        self.collection_probability = {}  # P(w|C)
        self.probabilities = {}  # Map of word probabilities

    def preprocess_lm(self, doc):
        text = doc.original_text

        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]  # Keep stopwords  
        tokens = [self.preprocessor.lemmatizer.lemmatize(word) for word in tokens]  # Preserve word forms  

        tokens.extend(self.preprocessor.extract_named_entities(text))  # Keep named entities for context  

        return " ".join(tokens)
    
    def preprocess_query(self, text):
        
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]  # Keep stopwords  
        tokens = [self.preprocessor.lemmatizer.lemmatize(word) for word in tokens]  # Preserve word forms  
        
        return " ".join(tokens)

    def build_index(self, documents):
        """
        Build the Language Model index by computing document term frequencies and collection probabilities.
        """
        if not documents:
            raise ValueError("No documents loaded. Use `load_documents()` first.")

        self.documents = copy.deepcopy(documents)
        for doc in self.documents:
            doc.preprocessed_text = self.preprocess_lm(doc)

        self.total_terms = sum(len(doc.preprocessed_text.split()) for doc in self.documents)
        self.doc_lengths = [len(doc.preprocessed_text.split()) for doc in self.documents]

        # Initialize the document frequencies
        for doc in self.documents:
            self.doc_frequencies[doc.doc_id] = Counter()  # Initialize empty Counter for each document
            tokens = doc.preprocessed_text.split()
            for term in tokens:
                self.doc_frequencies[doc.doc_id][term] += 1
                self.term_frequencies[term] += 1

        # Compute collection probability P(w|C)
        self.collection_probability = {
            term: self.term_frequencies[term] / self.total_terms for term in self.term_frequencies
        }

        # Store probabilities in a map (like the model file in the pseudocode)
        self.probabilities = self.collection_probability.copy()

    def compute_lm_entropy_and_coverage(self, query):
        """
        Compute the entropy and coverage of the given query using the language model.
        """
        words = word_tokenize(query.lower())
        total_words = len(words)

        H = 0  # Entropy
        unk_count = 0  # Unknown word count

        for word in words:
            P = self.lambda_unk / len(self.probabilities)  # Default probability for unknown words
            if word in self.probabilities:
                P += (1 - self.lambda_unk) * self.probabilities[word]  # Apply known word probability
            else:
                unk_count += 1  # Word was not found in probabilities

            H += -math.log2(P)  # Compute entropy contribution

        entropy = H / total_words
        coverage = (total_words - unk_count) / total_words

        return entropy, coverage

    def search(self, query):
        """
        Search for the query in the document collection using the Language Model.
        Returns ranked results with filename, filepath, similarity score, and snippet.
        """
        if not self.probabilities:
            raise ValueError("Language Model index not built. Load documents and build the index first.")

        processed_query = self.preprocess_query(query.query_name)
        entropy, coverage = self.compute_lm_entropy_and_coverage(processed_query)

        scores = [(idx, self.compute_lm_score(processed_query, idx)) for idx in range(len(self.tfidf_builder.documents))]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        all_results = []
        for idx, score in scores:
            doc = self.documents[idx]
            snippet = self.generate_snippet(doc.original_text, query.query_name.split())
            all_results.append({
                "doc_id": doc.doc_id,
                "file_name": doc.file_name,
                "path": doc.path,
                "original_text": doc.original_text,
                "score": score,
                "snippet": snippet,
                "extension": doc.file_extension,
                "bibliography": doc.bibliography,
                "author": doc.author
            })

        self.trec.save_to_trec(query, all_results)
        return all_results

    def compute_lm_score(self, query, doc_idx):
        """
        Compute the Language Model score for a given query and document index using Dirichlet smoothing.
        """
        query_terms = word_tokenize(query.lower())
        score = 0

        for term in query_terms:
            doc_id = self.documents[doc_idx].doc_id
            doc_term_freq = self.doc_frequencies.get(doc_id).get(term, 0)  # Use the updated doc_frequencies
            unk = self.lambda_unk / len(self.probabilities)
            collection_prob = self.collection_probability.get(term, unk)

            # Dirichlet smoothing formula
            term_probability = (doc_term_freq + self.mu * collection_prob) / (self.doc_lengths[doc_idx] + self.mu)
            score += math.log(term_probability)

        return score

    def generate_snippet(self, content, query_terms, snippet_length=30):
        """
        Generate a snippet from the document containing the query terms,
        with surrounding context and highlighted terms.
        """
        tokens = content.split()
        query_terms_lower = [term.lower() for term in query_terms]
        query_indices = [i for i, token in enumerate(tokens) if token.lower() in query_terms_lower]

        if not query_indices:
            return "No relevant snippet found."

        # Generate snippet around the first matching term
        start_index = max(0, query_indices[0] - snippet_length // 2)
        end_index = min(len(tokens), query_indices[0] + snippet_length // 2)

        snippet = " ".join(tokens[start_index:end_index])
        return snippet + "..." if end_index < len(tokens) else snippet