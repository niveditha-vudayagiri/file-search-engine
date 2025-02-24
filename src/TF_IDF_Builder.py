from sklearn.feature_extraction.text import TfidfVectorizer
from Document import Document
import os
import hashlib
import xml.etree.ElementTree as ET
import numpy as np

class TF_IDF_Builder:
    def __init__(self, preprocessor):
        """
        Initialize the TF-IDF builder.
        :param preprocessor: Preprocessing object to clean and preprocess text.
        """
        self.preprocessor = preprocessor
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", use_idf=True)
        self.documents = []
        self.tfidf_matrix = None

    def generate_doc_id(self,filename):
        return hashlib.md5(filename.encode()).hexdigest()

    def load_cranfield_xml(self,filepath):
        documents = []
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                xml_content = f"<root>\n{f.read()}</root>"  # Wrap docs in a fake root

            root = ET.fromstring(xml_content)  # Parse as a single XML structure

            for doc in root.findall("doc"):
                try:
                    doc_id = doc.find("docno").text.strip()
                    title = doc.find("title").text.strip() if doc.find("title").text is not None else ""
                    author = doc.find("author").text.strip() if doc.find("author").text is not None else ""
                    bib = doc.find("bib").text.strip() if doc.find("bib").text is not None else ""
                    text = doc.find("text").text.strip() if doc.find("text").text is not None else ""

                    documents.append({
                        "doc_id": doc_id,
                        "file_name": title,
                        "author": author,
                        "bibliography": bib,
                        "text": text
                    })

                except AttributeError as e:
                    print(f"Skipping a document due to missing fields: {e}")

        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
        except ET.ParseError as e:
            print(f"XML Parsing Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

        return documents
    def load_documents(self, folder_path):
        """
        Load all .txt documents from the given folder, preprocess them,
        and store their content in the `documents` list.
        """

        self.documents = []
        for idx, file in enumerate(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file)
            # Check if the file is a .txt file and if it exists
            """if os.path.isfile(file_path) and file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    preprocessed_text = self.preprocessor.preprocess(text)
                    file_extension = os.path.splitext(file)[1]  # Get the file extension
                    doc_id = self.generate_doc_id(file)
                    document = Document(doc_id,file, file_path, text, preprocessed_text, file_extension)
                    self.documents.append(document)"""
            # Or read and parse a Cranfield collection in TREC XML format
            if os.path.isfile(file_path) and file.startswith("cran.all"):
                documents = self.load_cranfield_xml(file_path)
                for doc in documents:
                    #preprocessed_text = self.preprocessor.preprocess(doc["text"])
                    #combine text, title, author, and bib and store in preprocessed_text
                    preprocessed_text = doc["text"]
                    document = Document(doc["doc_id"], doc["file_name"], file_path, doc["text"], preprocessed_text, ".xml",
                                        author=doc["author"], bibliography=doc["bibliography"])
                    self.documents.append(document)
            
        if not self.documents:
            raise ValueError("No .txt files found in the specified folder.")
        return self.documents

    def build_index(self, documents):
        """
        Build the TF-IDF index for the loaded documents.
        Complete pipeline: TF-IDF → EVSM → LSA
        """
        if not self.documents:
            raise ValueError("No documents loaded. Use `load_documents()` first.")
        
        """preprocessed_texts = [doc.preprocessed_text for doc in documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)

        """
        self.sentences = []  # Store sentence-level documents
        self.doc_mapping = []  # Track which sentence belongs to which document

        for doc_id, doc in enumerate(documents):
            self.sentences.extend(doc.sentences)  # Flatten the list of sentences
            self.doc_mapping.extend([doc_id] * len(doc.sentences))  # Map sentences to document IDs
        
        # Compute TF-IDF at sentence level
        self.tfidf_matrix = self.vectorizer.fit_transform(self.sentences)

        
    def get_query_vector(self, query):
        """
        Transform the query into a TF-IDF vector using the fitted vectorizer.
        """
        return self.vectorizer.transform([query])

    def get_tfidf_matrix(self):
        """
        Retrieve the TF-IDF matrix after building the index.
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF index not built. Use `build_index()` first.")
        return self.tfidf_matrix