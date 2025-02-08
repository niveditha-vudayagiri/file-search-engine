from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TextPreprocessor import TextPreprocessor
from utils import TRECUtilities
from Query import Query

class VectorSpaceModel:
    def __init__(self,tfidf_builder, trec):
        """
        Initialize the Vector Space Model.
        :param tfidf_builder: An instance of the TF_IDF_Builder class.
        """
        self.preprocessor = TextPreprocessor()
        self.tfidf_builder = tfidf_builder
        self.tfidf_matrix = None
        self.trec = trec

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

    def search(self, query):
        """
        Search for the query in the document collection.
        Returns ranked results with filename, filepath, similarity score, and snippet.
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF index not built. Load documents and build the index first.")
        
        processed_query = self.tfidf_builder.preprocessor.preprocess(query.query_name, True)
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
                snippet = self.generate_snippet(doc.original_text, query.query_name.split())
                all_results.append({
                    "doc_id": doc.doc_id,
                    "file_name": doc.file_name,
                    "path": doc.path,
                    "original_text": doc.original_text,
                    "score": similarities[idx],
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