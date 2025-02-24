import json,os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, PhotoImage
from VectorSpaceModel import VectorSpaceModel
from BestMatching25 import BM25
from LanguageModel import MultinomialLanguageModel
from TextPreprocessor import TextPreprocessor
from TF_IDF_Builder import TF_IDF_Builder
from SearchLogger import SearchLogger
from utils import IconLoadUtilities
import threading, asyncio
from datetime import datetime
import xml.etree.ElementTree as ET
from Query import Query
from utils import TRECUtilities
import numpy as np

class SearchApp:
    def __init__(self, master):
        self.master = master
        self.tf_idf= TF_IDF_Builder(TextPreprocessor())
        self.vsm = VectorSpaceModel(self.tf_idf, TRECUtilities( "vsm_results.trec"))
        self.lm = MultinomialLanguageModel(self.tf_idf,TRECUtilities( "lm_results.trec"))
        self.bm25 = BM25(self.tf_idf,TRECUtilities( "bm25_results.trec"))
        self.folder_path = None
        self.current_page = 1
        self.txt_image = None
        self.utils = IconLoadUtilities(master)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logger = SearchLogger(f"Search_log_{timestamp}.log")
        self.queries = []
        self.results = None

        self.documents = []

        # Start on Page 1
        self.page1()

    def clear_frame(self):
        for widget in self.master.winfo_children():
            widget.destroy()

    def page1(self):
        self.clear_frame()
        self.master.title("Document Indexer")
        self.master.geometry("600x200")

        tk.Label(self.master, text="Select a folder to index documents").pack(pady=10)
        browse_button = tk.Button(self.master, text="Browse", command=self.browse_folder)
        browse_button.pack()

        self.folder_label = tk.Label(self.master, text="No folder selected", fg="grey")
        self.folder_label.pack(pady=10)

        build_button = tk.Button(self.master, text="Build Index", command=self.build_index)
        build_button.pack(pady=10)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path = folder
            self.folder_label.config(text=f"Selected: {folder}", fg="black")

    def build_index(self):
        if not self.folder_path:
            messagebox.showwarning("Warning", "Please select a folder first.")
            return

        try:
           
            self.documents = self.tf_idf.load_documents(self.folder_path)

             #Build VSM Index
            self.vsm.build_index(self.documents)

            #Build BM25 Index
            self.bm25.build_index(self.documents)

            #Build LM Index
            self.lm.build_index(self.documents)

            messagebox.showinfo("Success", "Index built successfully!")
            self.page2()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def page2(self):
        self.clear_frame()
        self.master.title("Search Documents")
        self.master.geometry("800x500")

        tk.Label(self.master, text="Enter your query:").pack(pady=10)
        self.query_entry = tk.Entry(self.master, width=50)
        self.query_entry.pack()

        search_button = tk.Button(self.master, text="Search", command=self.search_query_button_handler)
        search_button.pack(pady=10)

        search_button = tk.Button(self.master, text="Search from TREC queries", command=self.search_button_handler)
        search_button.pack(pady=10)

        # Results section split into two frames
        self.results_frame = tk.Frame(self.master)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Left pane for list of results
        self.results_list_frame = tk.Frame(self.results_frame, width=300)
        self.results_list_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.results_tree = ttk.Treeview(self.results_list_frame, columns=("Filename"), show="tree")
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_tree.bind("<<TreeviewSelect>>", self.show_details)

        scrollbar = tk.Scrollbar(self.results_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_tree.yview)

        # Configure the Treeview with two columns: one for the icon and one for the filename
        self.results_tree["columns"] = ("filename",)
        self.results_tree.column("#0", width=40, stretch=tk.NO)  # Icon column with fixed width

        # Define the headings (optional if you want a header for the Treeview)
        self.results_tree.heading("#0", text="")  # No text for the icon column
        self.results_tree.heading("filename", text="Filename")

        # Right pane for detailed information
        self.details_frame = tk.Frame(self.results_frame)
        self.details_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.details_label = tk.Label(self.details_frame, text="Document Details", font=("Arial", 14, "bold"))
        self.details_label.pack(pady=10)

        self.metadata_label = tk.Label(self.details_frame, text="", anchor="w", justify=tk.LEFT)
        self.metadata_label.pack(fill=tk.BOTH, expand=True, padx=10)

        self.snippet_text = tk.Text(self.details_frame, wrap=tk.WORD, height=10)
        self.snippet_text.pack(fill=tk.BOTH, expand=True, padx=10)
        self.snippet_text.config(state=tk.DISABLED)

        # View Content
        self.view_content_button = tk.Button(
            self.details_frame, text="View Full Content", command=self.view_file_content
        )
        self.view_content_button.pack(pady=10)

        # Pagination
        self.pagination_frame = tk.Frame(self.master)
        self.pagination_frame.pack(pady=10)

        self.prev_button = tk.Button(self.pagination_frame, text="Previous", command=self.prev_page_handler, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.pagination_frame, text="Next", command=self.next_page_handler, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)

    def load_queries(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        for idx,top in enumerate(root.findall("top")):
            query_id = top.find("num").text.strip()
            query_text = top.find("title").text.strip()
            query_obj = Query(idx+1, query_text)
            self.queries.append(query_obj)
        
        return self.queries
    
    async def search_query(self,query):
        self.current_page = 1
        vsm_results = self.vsm.search(query)
        bm25_results = self.bm25.search(query)
        lm_results = self.lm.search(query)

        self.results = self.combine_results(vsm_results, bm25_results, lm_results)
        curr_results = self.get_curr_results(self.current_page, 15)

        query.add_result(curr_results)
        await self.display_results(query.results)

    async def search(self):

        #Read queries from folder_path+"cran.qry.xml" and split to queries
        self.queries = self.load_queries(self.folder_path+"/cran.qry.xml")
        for query in self.queries:
            await self.search_query(query)

    def combine_results(self, vsm_results, bm25_results, lm_results):
        """
        Combines results from VSM and BM25 into a single list for display.
        Ensures documents are merged based on doc_id, adding both scores.
        """
        combined_dict = {}

        # Add VSM results
        for vsm_result in vsm_results:
            doc_id = vsm_result["doc_id"]
            combined_dict[doc_id] = {
                "doc_id": doc_id,
                "file_name": vsm_result["file_name"],
                "path": vsm_result["path"],
                "original_text": vsm_result["original_text"],
                "extension": vsm_result["extension"],
                "vsm_score": vsm_result["score"],
                "bm25_score": 0,  # Default score if not in BM25 results
                "lm_score": 0,
                "snippet": vsm_result["snippet"],
                "bibliography": vsm_result["bibliography"],
                "author": vsm_result["author"]
            }

        # Merge BM25 results
        for bm25_result in bm25_results:
            doc_id = bm25_result["doc_id"]
            if doc_id in combined_dict:
                combined_dict[doc_id]["bm25_score"] = bm25_result["score"]
            else:
                combined_dict[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": bm25_result["file_name"],
                    "path": bm25_result["path"],
                    "original_text": bm25_result["original_text"],
                    "extension": bm25_result["extension"],
                    "vsm_score": 0,  # Default score if not in VSM results
                    "bm25_score": bm25_result["score"],
                    "lm_score": 0,
                    "snippet": bm25_result["snippet"],
                    "bibliography": bm25_result["bibliography"],
                    "author": bm25_result["author"]
                }

        # Merge LM results
        for lm_result in lm_results:
            doc_id = lm_result["doc_id"]
            if doc_id in combined_dict:
                combined_dict[doc_id]["lm_score"] = lm_result["score"]
            else:
                combined_dict[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": lm_result["file_name"],
                    "path": lm_result["path"],
                    "original_text": lm_result["original_text"],
                    "extension": lm_result["extension"],
                    "vsm_score": 0,  # Default score if not in VSM results
                    "bm25_score": 0,
                    "lm_score": lm_result["score"],
                    "snippet": lm_result["snippet"],
                    "bibliography": lm_result["bibliography"],
                    "author": lm_result["author"]
                }
        return list(combined_dict.values())
        
    
    def get_curr_results(self, page, results_per_page):
        # Pagination logic
        total_results = len(self.results)
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page

        paginated_results = self.results[start_index:end_index]
        has_next_page = end_index < total_results
        has_previous_page = start_index > 0

        return {
            "results": self.results,
            "paginated_results" : paginated_results,
            "has_previous_page": has_previous_page,
            "has_next_page": has_next_page
        }
        

    def load_icons_in_background(self, results, font_height):
        # Load the icon once (or as needed for different file types)
        if(self.txt_image == None):
            self.txt_image = self.utils.load_resized_icon("img/icons/txt.png", font_height)
        
        # Update the Treeview with icons in the main thread
        for i, result in enumerate(results["results"]):
            if result["file_name"].endswith(".txt"):
                # Use a lambda to delay the execution and pass the required arguments
                self.master.after(0, lambda idx=i: self.results_tree.item(idx, image=self.txt_image ))
                
    async def display_results(self, results):
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.metadata_label.config(text="")
        self.snippet_text.config(state=tk.NORMAL)
        self.snippet_text.delete("1.0", tk.END)
        self.snippet_text.config(state=tk.DISABLED)
        
        # Add results to the Treeview without icons initially
        self.current_results = results["paginated_results"] # Store results for details view
        for i, result in enumerate(self.current_results):
            self.results_tree.insert("", tk.END, iid=i, text="", 
            values=(
                result["file_name"], 
                f"VSM: {result['vsm_score']:.2f}, BM25: {result['bm25_score']:.2f}, LM: {result['lm_score']:.2f}"
            ))
    
        # Update pagination buttons
        self.prev_button.config(state=tk.NORMAL if results["has_previous_page"] else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if results["has_next_page"] else tk.DISABLED)

    def show_details(self, event):
        # Get selected item from Treeview
        selected_item = self.results_tree.selection()
        if not selected_item:
            return

        # Get the index of the selected document
        selected_index = int(selected_item[0])  # Treeview item ID corresponds to the index
        selected_doc = self.current_results[selected_index]

        # Display metadata
        metadata_text = (
            f"Title: {selected_doc['file_name']}\n"
            f"Author: {selected_doc['author']}\n"
            f"Bibliography: {selected_doc['bibliography']}\n"
            f"VSM Score: {selected_doc['vsm_score']:.2f}\n"
            f"BM25 Score: {selected_doc['bm25_score']:.2f}\n"
            f"Language Model Score: {selected_doc['lm_score']:.2f}"
        )
        self.metadata_label.config(text=metadata_text)

        # Display and highlight snippet
        snippet = selected_doc["snippet"]
        self.snippet_text.config(state=tk.NORMAL)
        self.snippet_text.delete("1.0", tk.END)
        self.snippet_text.insert(tk.END, snippet)

        # Highlight search term
        """start_idx = "1.0"
        while True:
            start_idx = self.snippet_text.search(self.query, start_idx, stopindex=tk.END, nocase=True)
            if not start_idx:
                break
            end_idx = f"{start_idx}+{len(self.query)}c"
            self.snippet_text.tag_add("highlight", start_idx, end_idx)
            start_idx = end_idx

        self.snippet_text.tag_config("highlight", background="yellow", foreground="black")
        self.snippet_text.config(state=tk.DISABLED)"""

    def prev_page_handler(self):
        asyncio.run(self.prev_page())

    async def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            curr_results = self.get_curr_results(self.current_page, 15)
            await self.display_results(curr_results)

    def next_page_handler(self):
        asyncio.run(self.next_page())

    async def next_page(self):
        self.current_page += 1
        curr_results = self.get_curr_results(self.current_page, 15)
        await self.display_results(curr_results)

    def search_query_button_handler(self):
        asyncio.run(self.search_query(self.query_entry.get().strip().lower()))  

    def search_button_handler(self):
        asyncio.run(self.search())
    
    def view_file_content(self):
        # Get selected item from Treeview
        selected_item = self.results_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a document first.")
            return

        # Get the index of the selected document
        selected_index = int(selected_item[0])  # Treeview item ID corresponds to the index
        selected_doc = self.current_results[selected_index]

        # Log the interaction for the query and document
        query = self.query_entry.get().strip().lower()
        doc_id = selected_doc["doc_id"]  # Unique identifier for the document
        self.log_interaction(query, doc_id)

        # Check if the file exists
        file_path = selected_doc["path"]
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"The file does not exist: {file_path}")
            return
        
         # Log the document opening
        self.logger.log_click(self.query, selected_doc["doc_id"], selected_doc["file_name"])

        # Create a new popup window to display content
        popup = tk.Toplevel(self.master)
        popup.title(f"Viewing: {selected_doc['file_name']}")
        popup.geometry("600x400")

        # Bind close event to log the time spent
        popup.protocol("WM_DELETE_WINDOW", lambda: self.close_document(selected_doc, popup))

        # Add a scrollable Text widget to display file content
        text_widget = tk.Text(popup, wrap=tk.WORD)
        text_widget.insert("1.0", selected_doc["original_text"])
        text_widget.config(state=tk.DISABLED)  # Make content read-only
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Add a scrollbar
        scrollbar = tk.Scrollbar(popup, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)

    def log_interaction(self, query, doc_id):
        # Load interaction data (from memory or file)
        try:
            with open("interaction_data.json", "r") as f:
                self.interaction_data = json.load(f)
        except FileNotFoundError:
            self.interaction_data = {}

        # Update interaction data
        if query not in self.interaction_data:
            self.interaction_data[query] = {}
        if doc_id not in self.interaction_data[query]:
            self.interaction_data[query][doc_id] = {"views": 0, "score_boost": 0.0}

        self.interaction_data[query][doc_id]["views"] += 1
        self.interaction_data[query][doc_id]["score_boost"] += 0.1  # Increment boost

        # Save interaction data back to file
        with open("interaction_data.json", "w") as f:
            json.dump(self.interaction_data, f)

    def close_document(self, selected_doc, popup):
        """Logs when a document is closed and records time spent."""
        doc_id = selected_doc["doc_id"]

        if doc_id in self.logger.opened_docs:
            self.logger.log_close(self.query, doc_id, selected_doc["file_name"])

        popup.destroy()  # Close the pop-up window


# Main Program
if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()