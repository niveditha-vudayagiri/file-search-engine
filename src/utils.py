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
    
        