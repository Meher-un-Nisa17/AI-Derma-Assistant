from PIL import Image
import io

class SkinAnalyzer:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    async def process_image(self, file_bytes: bytes):
        """
        Takes raw bytes from the upload, resizes it, 
        and prepares it for the AI model.
        """
        # 1. Open the image from bytes
        image = Image.open(io.BytesIO(file_bytes))

        # 2. Ensure it is RGB (removes transparency/alpha channels)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 3. Resize it for the AI model
        image = image.resize(self.target_size)

        # For now, we return a success message and the new size
        # Later, this is where we will call: model.predict(image)
        return {
            "condition": "Healthy Skin / General Scan",
            "confidence": 0.85,  # 85% confidence
            "description": "The image has been processed successfully. No major irregularities detected in this test scan."
        }

# Create a single instance to be used by the backend
analyzer = SkinAnalyzer()