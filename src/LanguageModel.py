import math
from collections import Counter
from TextPreprocessor import TextPreprocessor
from nltk.tokenize import word_tokenize

class MultinomialLanguageModel:
    def __init__(self, tfidf_builder, trec, mu=2000):
        """
        Initialize the Language Model for Information Retrieval.
        :param tfidf_builder: An instance of the TF_IDF_Builder class.
        :param mu: Dirichlet smoothing parameter.
        """
        self.preprocessor = tfidf_builder.preprocessor
        self.tfidf_builder = tfidf_builder
        self.trec = trec
        self.mu = mu

        self.total_terms = 0
        self.term_frequencies = Counter()
        self.doc_lengths = []
        self.collection_probability = {}

    def build_index(self, documents):
        """
        Build the Language Model index by computing document term frequencies and collection probabilities.
        """
        if not documents:
            raise ValueError("No documents loaded. Use `load_documents()` first.")

        self.doc_lengths = [len(doc.preprocessed_text.split()) for doc in documents]
        self.total_terms = sum(self.doc_lengths)

        # Compute term frequencies for each document and overall collection
        for doc in self.tfidf_builder.documents:
            tokens = doc.preprocessed_text.split()
            doc_term_counts = Counter(tokens)
            for term, count in doc_term_counts.items():
                self.term_frequencies[term] += count

        # Compute collection probability P(w|C) for each term
        self.collection_probability = {
            term: self.term_frequencies[term] / self.total_terms for term in self.term_frequencies
        }

    def preprocess_lm(self, text, isQuery=False):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]  # Keep stopwords  
        tokens = [self.preprocessor.lemmatizer.lemmatize(word) for word in tokens]  # Preserve word forms  

        tokens.extend(self.preprocessor.extract_named_entities(text))  # Keep named entities for context  

        return " ".join(tokens)
    
    def compute_lm_score(self, query, doc_idx):
        """
        Compute the Language Model score for a given query and document index.
        Uses Dirichlet smoothing.
        """
        query_terms = query.split()
        doc = self.tfidf_builder.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        term_frequencies = Counter(doc.preprocessed_text.split())

        score = 0
        for term in query_terms:
            doc_term_freq = term_frequencies.get(term, 0)
            collection_prob = self.collection_probability.get(term, 1e-10)  # Small value for unseen words

            # Dirichlet smoothing formula
            term_probability = (doc_term_freq + self.mu * collection_prob) / (doc_length + self.mu)
            score += math.log(term_probability)

        return score

    def search(self, query):
        """
        Search for the query in the document collection using the Language Model.
        Returns ranked results with filename, filepath, similarity score, and snippet.
        """
        if not self.doc_lengths:
            raise ValueError("Language Model index not built. Load documents and build the index first.")

        processed_query = self.preprocess_lm(query.query_name, True)
        scores = [(idx, self.compute_lm_score(processed_query, idx)) for idx in range(len(self.tfidf_builder.documents))]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        all_results = []
        for idx, score in scores:
            doc = self.tfidf_builder.documents[idx]
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

    def generate_snippet(self, content, query_terms, snippet_length=30):
        """
        Generate a snippet from the document containing the query terms,
        with surrounding context and highlighted terms.
        """
        tokens = content.split()
        query_terms_lower = [term.lower() for term in query_terms]
        query_indices = [
            i for i, token in enumerate(tokens)
            if token.lower() in query_terms_lower
        ]

        if not query_indices:
            return "No relevant snippet found."

        # Generate snippet around the first matching term
        start_index = max(0, query_indices[0] - snippet_length // 2)
        end_index = min(len(tokens), query_indices[0] + snippet_length // 2)

        snippet = " ".join(tokens[start_index:end_index])
        return snippet + "..." if end_index < len(tokens) else snippet