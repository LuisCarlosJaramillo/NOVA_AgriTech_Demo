import gradio as gr

def procesar_texto(entrada):
    salida = f"Procesaste: {entrada}"
    return salida

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Mi Demo IA")
    entrada = gr.Textbox(label="Escribe algo")
    salida = gr.Textbox(label="Resultado")
    btn = gr.Button("Procesar")
    btn.click(fn=procesar_texto, inputs=entrada, outputs=salida)

if __name__ == "__main__":
    demo.launch()