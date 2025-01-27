from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Document import Document
import os

class VectorSpaceModel:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.documents = []

    def load_documents(self, folder_path):
        """
        Load all .txt documents from the given folder, preprocess them,
        and store their content in the `documents` dictionary.
        """
        self.documents = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        preprocessed_text = self.preprocessor.preprocess(text)
                        file_extension = os.path.splitext(file)[1]  # Get the file extension
                        document = Document(file, file_path, text, preprocessed_text, file_extension)
                        self.documents.append(document)
        if not self.documents:
            raise ValueError("No .txt files found in the specified folder.")
        return self.documents

    def build_index(self):
        """
        Build the TF-IDF index for the loaded documents.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Use `load_documents()` first.")
        
        preprocessed_texts = [doc_data.preprocessed_text for doc_data in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)

    def search(self, query, page = 1, results_per_page= 12):
        """
        Search for the query in the document collection.
        Returns ranked results with filename, filepath, similarity score, and snippet.
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF index not built. Load documents and build the index first.")
        
        processed_query = self.preprocessor.preprocess(query)
        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Rank documents by similarity scores
        ranked_indices = similarities.argsort()[::-1]

        all_results = []
        for idx in ranked_indices:
            score = similarities[idx]
            if score > 0:  # Include only documents with non-zero similarity
                # Match the index to the corresponding document
                doc= self.documents[idx]
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
        # Highlight query terms
        for term in query_terms_lower:
            snippet = snippet.replace(term, f"<b>{term}</b>")
        return snippet + "..." if end_index < len(tokens) else snippet