import logging
import os
import time
from datetime import datetime

class SearchLogger:
    def __init__(self, log_file="search_log.txt", console_output=True):
        self.log_file = log_file
        self.console_output = console_output
        self.opened_docs = {}  # To store timestamps when a document is opened

        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Configure logging
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def log_query(self, query):
        """Logs when a user performs a search query."""
        log_message = f"SEARCH: User queried -> '{query}'"
        logging.info(log_message)
        if self.console_output:
            print(log_message)

    def log_click(self, query, doc_id, filename):
        """Logs when a user clicks to view a document and starts tracking time."""
        timestamp = time.time()  # Store the opening time
        self.opened_docs[doc_id] = timestamp

        log_message = f"CLICK: Query '{query}' -> Opened document [{doc_id}] {filename}"
        logging.info(log_message)
        if self.console_output:
            print(log_message)

    def log_close(self, query, doc_id, filename):
        """Logs when a user closes a document and records time spent."""
        if doc_id not in self.opened_docs:
            return  # If the document wasn't tracked, skip

        open_time = self.opened_docs.pop(doc_id)
        time_spent = round(time.time() - open_time, 2)  # Calculate duration

        log_message = f"CLOSE: Query '{query}' -> Closed document [{doc_id}] {filename} | Time spent: {time_spent} sec"
        logging.info(log_message)
        if self.console_output:
            print(log_message)