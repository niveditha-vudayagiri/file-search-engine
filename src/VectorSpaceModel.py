from collections import Counter
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from TextPreprocessor import TextPreprocessor
import numpy as np
from sklearn.decomposition import TruncatedSVD
import copy

class VectorSpaceModel:
    def __init__(self, tfidf_builder, trec):
        """
        Initialize the Vector Space Model.
        :param tfidf_builder: An instance of the TF_IDF_Builder class.
        :param trec: Instance for handling TREC-style evaluation.
        """
        self.preprocessor = tfidf_builder.preprocessor
        self.tfidf_builder = tfidf_builder
        self.tfidf_matrix = None
        self.trec = trec
        self.svd = TruncatedSVD(n_components=300)  # LSA component
        self.evsm_matrix = None
        self.lsa_matrix = None
        self.tdw = None
        self.documents = None
        self.sentence_doc_map = {}  # Map sentence index → document
        self.sentences = []  # Store segmented sentences

    def build_index(self, documents):
        """
        Preprocess documents, perform segmentation and tokenization, and build the TF-IDF index.
        """
        if not documents:
            raise ValueError("No documents loaded. Use `load_documents()` first.")
        
        self.documents = copy.deepcopy(documents)

        for doc_idx, doc in enumerate(self.documents):
            doc.preprocessed_text = self.preprocess_vsm(doc)  # Preprocess the document
            
            for sentence in doc.preprocessed_text:
                sentence = " ".join(sentence)
                self.sentence_doc_map[len(self.sentences)] = doc  # Link sentence to document
                self.sentences.append(sentence)
                doc.sentences.append(sentence)  # Store sentence-level text

        # Build TF-IDF only at the document level
        self.tfidf_builder.build_index(self.documents)
        self.tfidf_matrix = self.tfidf_builder.get_tfidf_matrix()

        # Compute Term Discrimination Weights (TDW)
        self.compute_tdw()
        self.apply_evsm_weights()
        self.apply_lsa()

    def preprocess_vsm(self, doc):
        """
        Perform segmentation and tokenization on a document.
        """
        text = doc.original_text
        text = self.preprocessor.case_insensitive(text)  # Convert to lowercase
        sentences = self.preprocessor.punkt_tokenize(text)  # Sentence segmentation
        cleaned = self.preprocessor.clean_text(sentences,True)
        processed_sentences = self.preprocessor.lemmatization(cleaned)
        processed_sentences.extend(self.preprocessor.extract_named_entities(text)) 
        doc.preprocessed_text = processed_sentences
        return processed_sentences  # Returns a list of tokenized sentences

    def compute_tdw(self):
        """Compute Term Discrimination Weight (TDW)"""
        term_variances = np.var(self.tfidf_matrix.toarray(), axis=0)
        self.tdw = np.sqrt(term_variances)  # TDW based on variance

    def apply_evsm_weights(self):
        """Modify TF-IDF weights using EVSM adjustments"""
        self.evsm_matrix = self.tfidf_matrix.multiply(self.tdw)

    def apply_lsa(self):
        """Apply LSA on EVSM-adjusted TF-IDF."""
        self.lsa_matrix = self.svd.fit_transform(self.evsm_matrix)

    def preprocess_query(self, text):
        """
        Preprocesses the query by applying case normalization, segmentation, and tokenization.
        """
        text = self.preprocessor.case_insensitive(text)  # Lowercase
        sentences = self.preprocessor.punkt_tokenize(text)  # Sentence segmentation
        cleaned = self.preprocessor.clean_text(sentences,False)
        processed_sentence = self.preprocessor.lemmatization(cleaned)

        processed_sentence.extend([self.preprocessor.synonym_expansion(word) for word in processed_sentence[0]])  # Can improve query recall

        return " ".join(processed_sentence[0])

    def search(self, query):
        """
        Search for the query in the document collection using VSM with optional PRF.
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF index not built. Load documents and build the index first.")
        
        query.query_name = self.preprocess_query(query.query_name)  # Preprocess query
        query_vector = self.transform_query(query.query_name)  

        similarities = cosine_similarity(query_vector, self.lsa_matrix).flatten()
        ranked_indices = similarities.argsort()[::-1]

        all_results = []
        docs_added = []
        for idx in ranked_indices:
            score = similarities[idx]
            if score > 0:
                doc = self.sentence_doc_map[idx]
                snippet = self.generate_snippet(doc.original_text, query.query_name.split())
                #Append to all_results if doc_id is not already in all_results using boolean array
                if doc.doc_id not in docs_added:
                    all_results.append({
                        "doc_id": doc.doc_id,
                        "file_name": doc.file_name,
                        "original_text": doc.original_text,
                        "preprocessed_text": doc.preprocessed_text,
                        "path": doc.path,
                        "score": score,
                        "snippet": snippet,
                        "extension": doc.file_extension,
                        "bibliography": doc.bibliography,
                        "author": doc.author
                    })
                    docs_added.append(doc.doc_id)

        self.trec.save_to_trec(query, all_results)

        return all_results

    def transform_query(self, query):
        """
        Transform a query using the same pipeline (TF-IDF → EVSM → LSA), but with short-query weighting.
        """
        query_tfidf = self.tfidf_builder.get_query_vector(query)
        query_length = len(query.split())
        query_weight_factor = 1.0 if query_length > 3 else 1.5  # Boost short queries

        query_evsm = query_tfidf.multiply(self.tdw * query_weight_factor)
        query_lsa = self.svd.transform(query_evsm)

        return query_lsa

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
        
        start_index = max(0, query_indices[0] - snippet_length // 2)
        end_index = min(len(tokens), query_indices[0] + snippet_length // 2)
        
        snippet = " ".join(tokens[start_index:end_index])
        return snippet + "..." if end_index < len(tokens) else snippet