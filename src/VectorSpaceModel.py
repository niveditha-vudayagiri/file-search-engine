from sklearn.metrics.pairwise import cosine_similarity
from TextPreprocessor import TextPreprocessor
import numpy as np
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize

class VectorSpaceModel:
    def __init__(self,tfidf_builder, trec):
        """
        Initialize the Vector Space Model.
        :param tfidf_builder: An instance of the TF_IDF_Builder class.
        """
        self.preprocessor = tfidf_builder.preprocessor
        self.tfidf_builder = tfidf_builder
        self.tfidf_matrix = None
        self.trec = trec
        self.svd = TruncatedSVD(n_components=100)  # LSA component
        self.evsm_matrix = None
        self.lsa_matrix = None
        self.tdw = None


    def compute_tdw(self):
        """Compute Term Discrimination Weight (TDW)"""
        term_variances = np.var(self.tfidf_matrix.toarray(), axis=0)
        self.tdw = np.sqrt(term_variances)  # TDW is based on variance

    def apply_evsm_weights(self):
        """Modify TF-IDF weights using EVSM adjustments"""
        self.evsm_matrix = self.tfidf_matrix.multiply(self.tdw)

    def get_evsm_matrix(self):
        return self.evsm_matrix
    
    def apply_lsa(self):
        """Apply LSA on EVSM-adjusted TF-IDF."""
        self.lsa_matrix = self.svd.fit_transform(self.evsm_matrix)

    def build_index(self, documents):
        """
        Build the TF-IDF index using the TF_IDF_Builder.
        """
        if not documents:
            raise ValueError("No documents loaded. Use `load_documents()` first.")
        for doc in documents:
            doc.preprocessed_text = self.preprocess_vsm(doc.original_text, False)

        self.tfidf_builder.build_index()
        self.tfidf_matrix = self.tfidf_builder.get_tfidf_matrix()
        self.compute_tdw()
        self.apply_evsm_weights()
        self.apply_lsa()

    def preprocess_vsm(self, text, isQuery=False):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in self.preprocessor.stop_words]
        
        tokens = [self.preprocessor.lemmatizer.lemmatize(word) for word in tokens]  # Prefer lemmatization
        if isQuery:
            tokens.extend([self.preprocessor.synonym_expansion(word) for word in tokens])  # Synonyms

        tokens.extend(self.preprocessor.extract_named_entities(text))  
        tokens.extend(self.preprocessor.generate_ngrams(text, 2))  # Bigrams
        tokens.extend(self.preprocessor.generate_ngrams(text, 3))  # Trigrams

        return " ".join(tokens)

    def search(self, query):
        """
        Search for the query in the document collection.
        Returns ranked results with filename, filepath, similarity score, and snippet.
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF index not built. Load documents and build the index first.")
        
        processed_query = self.preprocess_vsm(query.query_name, True)
        query_vector = self.transform_query(processed_query)
        similarities = cosine_similarity(query_vector, self.lsa_matrix).flatten()

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

    def transform_query(self, query):
        """Transform a query using the same pipeline (TF-IDF → EVSM → LSA)."""
        query_tfidf = self.tfidf_builder.get_query_vector(query)
        query_evsm = query_tfidf.multiply(self.tdw)
        query_lsa = self.svd.transform(query_evsm)
        return query_lsa