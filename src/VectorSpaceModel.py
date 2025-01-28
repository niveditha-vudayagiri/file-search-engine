from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TextPreprocessor import TextPreprocessor
import os

class VectorSpaceModel:
    def __init__(self,tfidf_builder):
        """
        Initialize the Vector Space Model.
        :param tfidf_builder: An instance of the TF_IDF_Builder class.
        """
        self.preprocessor = TextPreprocessor()
        self.tfidf_builder = tfidf_builder
        self.tfidf_matrix = None

    def load_documents(self, folder_path):
        """
        Load documents using the TF_IDF_Builder.
        """
        return self.tfidf_builder.load_documents(folder_path)

    def build_index(self):
        """
        Build the TF-IDF index using the TF_IDF_Builder.
        """
        self.tfidf_builder.build_index()
        self.tfidf_matrix = self.tfidf_builder.get_tfidf_matrix()

    def search(self, query, page = 1, results_per_page= 12):
        """
        Search for the query in the document collection.
        Returns ranked results with filename, filepath, similarity score, and snippet.
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF index not built. Load documents and build the index first.")
        
        processed_query = self.tfidf_builder.preprocessor.preprocess(query)
        query_vector = self.tfidf_builder.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Rank documents by similarity scores
        ranked_indices = similarities.argsort()[::-1]

        all_results = []
        for idx in ranked_indices:
            score = similarities[idx]
            if score > 0:  # Include only documents with non-zero similarity
                # Match the index to the corresponding document
                doc= self.tfidf_builder.documents[idx]
                snippet = self.generate_snippet(doc.original_text, query.split())
                all_results.append({
                    "file_name": doc.file_name,
                    "path": doc.path,
                    "score": similarities[idx],
                    "snippet": snippet,
                    "extension": doc.file_extension,
                    "date": doc.date
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
            "results_per_page": results_per_page
        }
        
        return results

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