from datetime import datetime

class Document:
    def __init__(self, doc_id, file_name, path, original_text, preprocessed_text, file_extension,  author=None,
                 bibliography=None):
        self.doc_id = doc_id
        self.file_name = file_name
        self.path = path
        self.original_text = original_text
        self.preprocessed_text = preprocessed_text
        self.file_extension = file_extension
        self.author = author
        self.bibliography = bibliography
             
