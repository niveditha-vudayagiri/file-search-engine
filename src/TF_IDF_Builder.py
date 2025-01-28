from sklearn.feature_extraction.text import TfidfVectorizer
from Document import Document
import os


class TF_IDF_Builder:
    def __init__(self, preprocessor):
        """
        Initialize the TF-IDF builder.
        :param preprocessor: Preprocessing object to clean and preprocess text.
        """
        self.preprocessor = preprocessor
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.tfidf_matrix = None

    def load_documents(self, folder_path):
        """
        Load all .txt documents from the given folder, preprocess them,
        and store their content in the `documents` list.
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
        
        preprocessed_texts = [doc.preprocessed_text for doc in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)

    def get_tfidf_matrix(self):
        """
        Retrieve the TF-IDF matrix after building the index.
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF index not built. Use `build_index()` first.")
        return self.tfidf_matrix