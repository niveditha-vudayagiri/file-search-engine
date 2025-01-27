import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from VectorSpaceModel import VectorSpaceModel
from TextPreprocessor import TextPreprocessor


class SearchApp:
    def __init__(self, master):
        self.master = master
        self.vsm = VectorSpaceModel(TextPreprocessor())
        self.folder_path = None
        self.current_page = 1
        self.query = ""

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
        self.master.geometry("700x500")

        tk.Label(self.master, text="Enter your query:").pack(pady=10)
        self.query_entry = tk.Entry(self.master, width=50)
        self.query_entry.pack()

        search_button = tk.Button(self.master, text="Search", command=self.search)
        search_button.pack(pady=10)

        self.results_frame = tk.Frame(self.master)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_table = ttk.Treeview(
            self.results_frame,
            columns=("Filename", "Filepath", "Score", "Extension", "Date", "Snippet"),
            show='headings',
            height=15
        )
        self.results_table.heading("Filename", text="Filename")
        self.results_table.heading("Filepath", text="Filepath")
        self.results_table.heading("Score", text="Score")
        self.results_table.heading("Extension", text="Extension")
        self.results_table.heading("Date", text="Date")
        self.results_table.heading("Snippet", text="Snippet")
        self.results_table.column("Filename", width=100)
        self.results_table.column("Filepath", width=200)
        self.results_table.column("Score", width=50)
        self.results_table.column("Extension", width=70)
        self.results_table.column("Date", width=100)
        self.results_table.column("Snippet", width=300)
        self.results_table.pack(fill=tk.BOTH, expand=True)

        self.pagination_frame = tk.Frame(self.master)
        self.pagination_frame.pack(pady=10)

        self.prev_button = tk.Button(self.pagination_frame, text="Previous", command=self.prev_page, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.pagination_frame, text="Next", command=self.next_page, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)

    def search(self):
        self.query = self.query_entry.get().strip()
        if not self.query:
            messagebox.showwarning("Warning", "Please enter a query.")
            return

        self.current_page = 1
        results = self.vsm.search(self.query, page=self.current_page)
        self.display_results(results)

    def display_results(self, results):
        # Clear previous results
        for row in self.results_table.get_children():
            self.results_table.delete(row)

        # Display results in the table
        for result in results["results"]:
            self.results_table.insert(
                "", "end",
                values=(
                    result["file_name"],
                    result["path"],
                    f"{result['score']:.2f}",
                    result["extension"],
                    result["date"].strftime("%Y-%m-%d %H:%M:%S"),
                    result["snippet"]
                )
            )

        # Handle pagination buttons
        self.prev_button.config(state=tk.NORMAL if results["has_previous_page"] else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if results["has_next_page"] else tk.DISABLED)

    def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            results = self.vsm.search(self.query, page=self.current_page)
            self.display_results(results)

    def next_page(self):
        self.current_page += 1
        results = self.vsm.search(self.query, page=self.current_page)
        self.display_results(results)


# Main Program
if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()