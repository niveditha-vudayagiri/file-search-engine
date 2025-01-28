import math
from collections import Counter
from TextPreprocessor import TextPreprocessor

class BM25:
    def __init__(self, tfidf_builder, k1=1.5, b=0.75):
        """
        Initialize the BM25 model.
        :param tfidf_builder: An instance of the TF_IDF_Builder class.
        :param k1: Term frequency saturation parameter.
        :param b: Length normalization parameter.
        """
        self.tfidf_builder = tfidf_builder
        self.k1 = k1
        self.b = b
        self.avg_doc_length = 0
        self.doc_lengths = []
        self.doc_frequencies = Counter()
        self.total_documents = 0

    def load_documents(self, folder_path):
        """
        Load documents using the TF_IDF_Builder.
        """
        return self.tfidf_builder.load_documents(folder_path)

    def build_index(self):
        """
        Build the BM25 index, calculating term frequencies and document lengths.
        """
        if not self.tfidf_builder.documents:
            raise ValueError("No documents loaded. Use `load_documents()` first.")

        self.total_documents = len(self.tfidf_builder.documents)
        self.doc_lengths = [len(doc.preprocessed_text.split()) for doc in self.tfidf_builder.documents]
        self.avg_doc_length = sum(self.doc_lengths) / self.total_documents

        # Calculate document frequencies for terms
        for doc in self.tfidf_builder.documents:
            unique_terms = set(doc.preprocessed_text.split())
            for term in unique_terms:
                self.doc_frequencies[term] += 1

    def compute_bm25_score(self, query, doc_idx):
        """
        Compute the BM25 score for a given query and document index.
        :param query: The query string.
        :param doc_idx: Index of the document in the loaded documents list.
        :return: BM25 score for the document.
        """
        query_terms = query.split()
        doc = self.tfidf_builder.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        term_frequencies = Counter(doc.preprocessed_text.split())

        score = 0
        for term in query_terms:
            if term in self.doc_frequencies:
                # BM25 components
                df = self.doc_frequencies[term]
                idf = math.log((self.total_documents - df + 0.5) / (df + 0.5) + 1)
                tf = term_frequencies[term]
                norm_tf = ((tf * (self.k1 + 1)) /
                           (tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))))
                score += idf * norm_tf

        return score

    def search(self, query, page=1, results_per_page=12):
        """
        Search for the query in the document collection using BM25 scoring.
        Returns ranked results with filename, filepath, similarity score, and snippet.
        """
        if not self.doc_lengths:
            raise ValueError("BM25 index not built. Load documents and build the index first.")

        query = self.tfidf_builder.preprocessor.preprocess(query)
        scores = [(idx, self.compute_bm25_score(query, idx))
                  for idx in range(len(self.tfidf_builder.documents))]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Collect results with pagination
        all_results = []
        for idx, score in scores:
            if score > 0:  # Include only documents with non-zero scores
                doc = self.tfidf_builder.documents[idx]
                snippet = self.generate_snippet(doc.original_text, query.split())
                all_results.append({
                    "file_name": doc.file_name,
                    "path": doc.path,
                    "score": score,
                    "snippet": snippet,
                    "extension": doc.file_extension,
                    "date": doc.date,
                })

        # Pagination logic
        total_results = len(all_results)
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page

        paginated_results = all_results[start_index:end_index]
        has_next_page = end_index < total_results
        has_previous_page = start_index > 0

        return {
            "results": paginated_results,
            "total_results": total_results,
            "current_page": page,
            "has_next_page": has_next_page,
            "has_previous_page": has_previous_page,
            "results_per_page": results_per_page,
        }

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