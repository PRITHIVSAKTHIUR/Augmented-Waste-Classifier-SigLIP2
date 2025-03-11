![awsdawd.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/mRNCplvIRLYqoqRT2JKpP.png)

# Augmented-Waste-Classifier-SigLIP2

> **Augmented-Waste-Classifier-SigLIP2** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify waste types using the **SiglipForImageClassification** architecture.  

```py
    Classification Report:
                  precision    recall  f1-score   support
    
         Battery     0.9987    0.9987    0.9987      3840
      Biological     0.9998    0.9960    0.9979      4036
       Cardboard     0.9956    0.9909    0.9932      3628
         Clothes     0.9957    0.9914    0.9935      5336
           Glass     0.9800    0.9914    0.9856      4048
           Metal     0.9892    0.9965    0.9929      3136
           Paper     0.9937    0.9891    0.9914      4308
         Plastic     0.9865    0.9798    0.9831      3568
           Shoes     0.9876    0.9990    0.9933      3990
           Trash     1.0000    0.9939    0.9970      2796
    
        accuracy                         0.9926     38686
       macro avg     0.9927    0.9927    0.9927     38686
    weighted avg     0.9926    0.9926    0.9926     38686
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/0lXpKNyqS0i8ZjTRr42gl.png)

The model categorizes images into 10 waste classes:

    Class 0: "Battery"
    Class 1: "Biological"
    Class 2: "Cardboard"
    Class 3: "Clothes"
    Class 4: "Glass"
    Class 5: "Metal"
    Class 6: "Paper"
    Class 7: "Plastic"
    Class 8: "Shoes"
    Class 9: "Trash"

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Augmented-Waste-Classifier-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def waste_classification(image):
    """Predicts waste classification for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Battery", "1": "Biological", "2": "Cardboard", "3": "Clothes", 
        "4": "Glass", "5": "Metal", "6": "Paper", "7": "Plastic", 
        "8": "Shoes", "9": "Trash"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=waste_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Augmented Waste Classification",
    description="Upload an image to classify the type of waste."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()

```
# Intended Use:

The **Augmented-Waste-Classifier-SigLIP2** model is designed to classify different types of waste based on images. Potential use cases include:  

- **Waste Management:** Identifying and categorizing waste materials for proper disposal.
- **Recycling Assistance:** Helping users determine recyclable materials.
- **Environmental Monitoring:** Automating waste classification for smart cities.
- **AI-Powered Sustainability Solutions:** Supporting AI-based waste sorting systems to improve recycling efficiency.
