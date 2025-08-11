import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageClassification

# Cargar el modelo y el procesador
MODEL_NAME = "LuisCarlosJaramillo/NOVA_AgriTech_Demo_Model"  # Cambia si el repo del modelo es otro
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

def predict(image):
    # Preprocesar la imagen
    inputs = processor(images=image, return_tensors="pt")

    # Realizar predicción
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
    description="Sube una imagen de un cultivo para detectar su estado o tipo. Entrenado con datos agrícolas."
)

if __name__ == "__main__":
    demo.launch()
