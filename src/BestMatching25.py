import json
import math
from collections import Counter
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import copy

class BM25:
    def __init__(self, tfidf_builder, trec,k1=1.5, b=0.75 ):
        """
        Initialize the BM25 model.
        :param tfidf_builder: An instance of the TF_IDF_Builder class.
        :param k1: Term frequency saturation parameter.
        :param b: Length normalization parameter.
        """
        self.preprocessor = tfidf_builder.preprocessor
        self.tfidf_builder = tfidf_builder

        self.k1 = k1
        self.b = b

        self.documents = []
        self.avg_doc_length = 0
        self.doc_lengths = []
        self.doc_frequencies = Counter()
        self.total_documents = 0
        self.trec = trec

    def build_index(self, documents):
        """
        Build the BM25 index, calculating term frequencies and document lengths.
        """
        if not documents:
            raise ValueError("No documents loaded. Use `load_documents()` first.")

        self.documents = copy.deepcopy(documents.copy())

        for doc in self.documents:
            doc.preprocessed_text = self.preprocess_bm25(doc)

        self.tokenized_corpus = [doc.preprocessed_text.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)  # Efficient BM25 indexing
    
    def preprocess_bm25(self, doc):
        text = doc.original_text

        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]  # Keep stopwords
        tokens = [self.preprocessor.lemmatizer.lemmatize(word) for word in tokens]  # Optional, keeps word forms

        tokens.extend(self.preprocessor.extract_named_entities(text))  

        return " ".join(tokens)
    
    def preprocess_query(self, text):

        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]  # Keep stopwords
        tokens = [self.preprocessor.lemmatizer.lemmatize(word) for word in tokens]  # Optional, keeps word forms

        tokens.extend([self.preprocessor.synonym_expansion(word) for word in tokens])  # Can improve query recall

        return " ".join(tokens)
      

    def search(self, query):
        """
        Search for the query in the document collection using BM25 scoring.
        Returns ranked results with filename, filepath, similarity score, and snippet.
        """
        if not self.bm25:
            raise ValueError("BM25 index not built. Load documents and build the index first.")

        processed_query = self.preprocess_query(query.query_name)
        query_terms = processed_query.split()

        scores = self.bm25.get_scores(query_terms)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Collect results with pagination
        all_results = []
        for idx in ranked_indices:
            if scores[idx] > 0:  # Include only documents with non-zero scores
                doc = self.documents[idx]
                snippet = self.generate_snippet(doc.original_text, query.query_name.split())
                all_results.append({
                    "doc_id": doc.doc_id,
                    "file_name": doc.file_name,
                    "original_text": doc.original_text,
                    "path": doc.path,
                    "score": scores[idx],
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