import gradio as gr
from transformers import pipeline

# Dein Modell laden
model = pipeline('text-generation', model="gpt2")

# Funktion für die Antwortgenerierung ohne Filter
def generate_response(input_text):
    # Antwort des Modells generieren mit optimierten Parametern und ohne Filter
    response = model(
        input_text, 
        max_length=200,  # Erhöht die Antwortlänge
        temperature=0,  # Deterministische Antworten (keine Zufallsantworten)
        top_k=50,  # Top-K Sampling
        top_p=0.95,  # Top-P Sampling
        no_repeat_ngram_size=2,  # Verhindert Wiederholung von n-Grammen
        do_sample=True  # Sampling aktivieren
    )[0]['generated_text']
    
    return response

# Gradio Interface erstellen
iface = gr.Interface(fn=generate_response, inputs="text", outputs="text", live=True)

# Interface starten
iface.launch()