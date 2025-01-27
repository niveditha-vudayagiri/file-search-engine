from datetime import datetime

class Document:
    def __init__(self, file_name, path, original_text, preprocessed_text, file_extension, date=None):
        self.file_name = file_name
        self.path = path
        self.original_text = original_text
        self.preprocessed_text = preprocessed_text
        self.file_extension = file_extension
        self.date = date or datetime.now()  # Default to current date if not provided
             
