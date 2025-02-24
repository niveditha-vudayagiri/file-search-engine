import asyncio
from PIL import Image, ImageTk

class IconLoadUtilities:
    def __init__(self,master):
        self.txt_image = None
        self.master = master

    # Add an async version of the image loading function
    async def load_resized_icon_async(self, icon_path, target_height):
        return await asyncio.to_thread(self.load_resized_icon, icon_path, target_height)

    # Existing synchronous function for reference
    def load_resized_icon(self, icon_path, target_height):
        img = Image.open(icon_path)
        # Calculate width while preserving aspect ratio
        aspect_ratio = img.width / img.height
        target_width = int(target_height * aspect_ratio)
        # Resize the image
        img = img.resize((target_width, target_height), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

class TRECUtilities:
    def __init__(self, output_path="output.trec"):
        self.query = None
        self.output_path = output_path
        # Clear the file upon initialization
        open(self.output_path, "w").close()

    def save_to_trec(self, query, results):
        with open(self.output_path, "a") as f:  # Append mode
            for i, result in enumerate(results[:100]):
                # query_id iter document_id rank similarity run_id
                f.write(f"{query.query_id} Q0 {result['doc_id']} {i + 1} {result['score']} STANDARD\n")
    
        