import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Lade das größere GPT-J Modell und Tokenizer
model_name = "EleutherAI/gpt-j-6B"  # GPT-J 6B Modell von EleutherAI
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Funktion zur Antwortgenerierung
conversation_history = []  # Liste für das Gedächtnis der Konversation

def generate_response(input_text):
    global conversation_history  # Zugriff auf das globale Gedächtnis
    
    try:
        # Wenn es eine Konversation gibt, füge die Eingabe und Antwort zum Gedächtnis hinzu
        conversation_history.append(f"User: {input_text}")
        
        # Kodieren der Eingabe
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Hier verwenden wir do_sample=True für zufälligere, kreativere Antworten
        outputs = model.generate(inputs,
                                 temperature=0.7,  # Etwas höhere Temperatur für mehr Kreativität
                                 max_length=500,   # Maximale Länge der Antwort
                                 top_k=50,         # Top-K Sampling
                                 top_p=0.95,       # Top-P Sampling
                                 no_repeat_ngram_size=3,  # Verhindert die Wiederholung von n-grams
                                 do_sample=True,   # Aktiviert Sampling für zufällige Antworten
                                 pad_token_id=tokenizer.eos_token_id)  # Kein Padding, nur Ende-Token

        # Die generierte Antwort dekodieren
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
                         title="GPT-J Konversationsbot", 
                         description="Ein GPT-J basierter Bot, der zufällige und kreative Antworten ohne Einschränkungen bietet.")

# Starte das Interface
interface.launch()
