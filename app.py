import gradio as gr
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification

# Ruta local al modelo en tu repo
LOCAL_MODEL_PATH = "./model"

# Carga del procesador y modelo desde carpeta local
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)

def predict(image):
    # Preprocesar imagen
    inputs = processor(images=image, return_tensors="pt")

    # Inferencia
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    label = model.config.id2label[predicted_class_idx]
    return f"Predicción: {label}"

# Interfaz Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌱 NOVA AgriTech - Clasificador de Cultivos",
    description="Sube una imagen de un cultivo para detectar su estado o tipo."
)

if __name__ == "__main__":
    demo.launch()
