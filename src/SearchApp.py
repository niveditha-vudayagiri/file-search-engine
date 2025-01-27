import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, PhotoImage
from VectorSpaceModel import VectorSpaceModel
from TextPreprocessor import TextPreprocessor
import asyncio
from utils import IconLoadUtilities
import threading

class SearchApp:
    def __init__(self, master):
        self.master = master
        self.vsm = VectorSpaceModel(TextPreprocessor())
        self.folder_path = None
        self.current_page = 1
        self.query = ""
        self.txt_image = None
        self.utils = IconLoadUtilities(master)

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
            self.vsm.load_documents(self.folder_path)
            self.vsm.build_index()
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

        search_button = tk.Button(self.master, text="Search", command=self.search_button_handler)
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

        # Pagination
        self.pagination_frame = tk.Frame(self.master)
        self.pagination_frame.pack(pady=10)

        self.prev_button = tk.Button(self.pagination_frame, text="Previous", command=self.prev_page, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.pagination_frame, text="Next", command=self.next_page, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)

    async def search(self):
        self.query = self.query_entry.get().strip()
        if not self.query:
            messagebox.showwarning("Warning", "Please enter a query.")
            return

        self.current_page = 1
        results = self.vsm.search(self.query, page=self.current_page)
        await self.display_results(results)

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

        # Assuming `txt_icon` is loaded earlier in the class or method
        font_height = 20 # Approximate font height (adjust based on your GUI)
        
        # Add results to the Treeview without icons initially
        self.current_results = results["results"]  # Store results for details view
        for i, result in enumerate(results["results"]):
            self.results_tree.insert("", tk.END, iid=i, text="", values=(result["file_name"],))

        # Load icons
        #self.load_icons_in_background(results, font_height)
        thread = threading.Thread(target=self.load_icons_in_background, args=(results, font_height))
        thread.daemon = True  # Ensures the thread exits when the main program does
        thread.start()
    
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
            f"Filename: {selected_doc['file_name']}\n"
            f"Path: {selected_doc['path']}\n"
            f"File Type: {selected_doc['extension']}\n"
            f"Date: {selected_doc['date']}\n"
            f"Score: {selected_doc['score']:.2f}"
        )
        self.metadata_label.config(text=metadata_text)

        # Display and highlight snippet
        snippet = selected_doc["snippet"]
        self.snippet_text.config(state=tk.NORMAL)
        self.snippet_text.delete("1.0", tk.END)
        self.snippet_text.insert(tk.END, snippet)

        # Highlight search term
        start_idx = "1.0"
        while True:
            start_idx = self.snippet_text.search(self.query, start_idx, stopindex=tk.END, nocase=True)
            if not start_idx:
                break
            end_idx = f"{start_idx}+{len(self.query)}c"
            self.snippet_text.tag_add("highlight", start_idx, end_idx)
            start_idx = end_idx

        self.snippet_text.tag_config("highlight", background="yellow", foreground="black")
        self.snippet_text.config(state=tk.DISABLED)

    async def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            results = self.vsm.search(self.query, page=self.current_page)
            await self.display_results(results)

    async def next_page(self):
        self.current_page += 1
        results = self.vsm.search(self.query, page=self.current_page)
        await self.display_results(results)

    def search_button_handler(self):
        asyncio.run(self.search())


# Main Program
if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()