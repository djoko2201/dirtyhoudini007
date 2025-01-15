import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator  # Importiere den Google Translate Übersetzer
import json

# Lade das größere GPT-J Modell und Tokenizer
model_name = "EleutherAI/gpt-j-6B"  # GPT-J 6B Modell von EleutherAI
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialisiere den Übersetzer
translator = Translator()

# Funktion zur Antwortgenerierung
conversation_history = []  # Liste für das Gedächtnis der Konversation

# Beispiel für die Einstellungen, die gespeichert werden sollen
settings = {
    "temperature": 0.7,
    "max_length": 500,
    "top_k": 50,
    "top_p": 0.95,
    "no_repeat_ngram_size": 3
}

def save_settings():
    try:
        with open("settings.json", "w") as f:
            json.dump(settings, f)
    except Exception as e:
        print(f"Fehler beim Speichern der Einstellungen: {str(e)}")

def load_settings():
    global settings
    try:
        with open("settings.json", "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Keine gespeicherten Einstellungen gefunden, Standardwerte werden verwendet.")
    except Exception as e:
        print(f"Fehler beim Laden der Einstellungen: {str(e)}")

def generate_response(input_text):
    global conversation_history  # Zugriff auf das globale Gedächtnis
    
    try:
        # Wenn es eine Konversation gibt, füge die Eingabe und Antwort zum Gedächtnis hinzu
        conversation_history.append(f"User: {input_text}")
        
        # Kodieren der Eingabe
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Hier verwenden wir do_sample=True für zufälligere, kreativere Antworten
        outputs = model.generate(inputs,
                                 temperature=settings["temperature"],  # Temperature aus den gespeicherten Einstellungen
                                 max_length=settings["max_length"],   # Maximale Länge der Antwort
                                 top_k=settings["top_k"],         # Top-K Sampling
                                 top_p=settings["top_p"],       # Top-P Sampling
                                 no_repeat_ngram_size=settings["no_repeat_ngram_size"],  # Verhindert die Wiederholung von n-grams
                                 do_sample=True,   # Aktiviert Sampling für zufällige Antworten
                                 pad_token_id=tokenizer.eos_token_id)  # Kein Padding, nur Ende-Token

        # Die generierte Antwort dekodieren
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Übersetze die Antwort ins Deutsche
        translated_response = translator.translate(response, src='en', dest='de').text

        # Füge die Antwort auch zum Gedächtnis hinzu
        conversation_history.append(f"Bot: {translated_response}")

        # Rückgabe der übersetzten Antwort
        return translated_response

    except Exception as e:
        # Fehlerbehandlung bei Problemen
        print(f"Fehler bei der Antwortgenerierung: {str(e)}")
        save_settings()  # Speichert die aktuellen Einstellungen, wenn ein Fehler auftritt
        return f"Fehler: {str(e)}"

# Gradio-Interface erstellen
interface = gr.Interface(fn=generate_response, 
                         inputs="text", 
                         outputs="text", 
                         live=True,  # Option für Live-Updates
                         title="GPT-J Konversationsbot", 
                         description="Ein GPT-J basierter Bot, der zufällige und kreative Antworten auf Deutsch gibt.")

# Lade gespeicherte Einstellungen
load_settings()

# Starte das Interface
interface.launch()