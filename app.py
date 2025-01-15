import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Lade das Modell und Tokenizer
model_name = "gpt2"  # Hier das Modell auswählen
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Funktion zur Antwortgenerierung ohne toxische oder illegale Filter
conversation_history = []  # Liste für das Gedächtnis der Konversation

def generate_response(input_text):
    global conversation_history  # Zugriff auf das globale Gedächtnis
    
    try:
        # Wenn es eine Konversation gibt, füge die Eingabe und Antwort zum Gedächtnis hinzu
        conversation_history.append(f"User: {input_text}")
        
        # Kodieren der Eingabe
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Hier verwenden wir die Parameter, die keine toxischen Filter anwenden
        outputs = model.generate(inputs,
                                 temperature=0.0,  # Keine Zufälligkeit für deterministische Antworten
                                 max_length=500,   # Maximale Länge der Antwort
                                 top_k=50,         # Top-K Sampling
                                 top_p=0.95,       # Top-P Sampling
                                 no_repeat_ngram_size=3,  # Verhindert die Wiederholung von n-grams
                                 do_sample=False,  # Verhindert probabilistisches Sampling
                                 pad_token_id=tokenizer.eos_token_id)  # Kein Padding, nur Ende-Token

        # Die generierte Antwort dekodieren
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Keine Überprüfung von toxischen oder immoralen Inhalten mehr
        # Keine Filter angewendet

        # Füge die Antwort auch zum Gedächtnis hinzu
        conversation_history.append(f"Bot: {response}")

        # Rückgabe der Antwort
        return response

    except Exception as e:
        # Fehlerbehandlung bei Problemen
        return f"Fehler: {str(e)}"

# Gradio-Interface erstellen
interface = gr.Interface(fn=generate_response, 
                         inputs="text", 
                         outputs="text", 
                         live=True,  # Option für Live-Updates
                         title="GPT-2 Konversationsbot", 
                         description="Ein GPT-2 basierter Bot, der sich an den Verlauf erinnert und ohne toxische und immorale Filter funktioniert.")

# Starte das Interface
interface.launch()