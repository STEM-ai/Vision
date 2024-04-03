from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'https://assets.wfcdn.com/im/16661612/compr-r85/4172/41722749/new-york-city-map-on-paper-print.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('STEM-AI-mtl/City_map-vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('STEM-AI-mtl/City_map-vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
