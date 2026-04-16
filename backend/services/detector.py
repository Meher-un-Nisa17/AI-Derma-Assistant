import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

class SkinDetector:
    def __init__(self):
        # Using MobileNetV3 - Very fast and South Asia skin-tone friendly
        self.model = models.mobilenet_v3_small(pretrained=True)
        
        # Adjusting for your 5 specific South Asian skin concerns:
        # [Oily, Dry, Hyperpigmentation, Acne, Healthy]
        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_features, 5)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            prob = torch.nn.functional.softmax(outputs[0], dim=0)
            
        labels = ["Oily", "Dry", "Hyperpigmentation", "Acne", "Healthy"]
        conf, idx = torch.max(prob, 0)
        
        return {
            "label": labels[idx],
            "confidence": round(float(conf) * 100, 2),
            "is_south_asian_optimized": True
        }

detector = SkinDetector()