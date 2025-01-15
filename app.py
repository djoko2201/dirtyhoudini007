import gradio as gr
from transformers import pipeline

# Modell laden (wir verwenden ein GPT-Modell als Beispiel)
chatbot = pipeline("text-generation", model="gpt2")

# Funktion zur Chat-Verarbeitung
def chat_with_bot(user_input, history=[]):
    # Der gesamte Verlauf der Konversation wird als Eingabekontext verwendet
    context = " ".join([f"User: {message[0]} Bot: {message[1]}" for message in history]) + f" User: {user_input}"
    
    # Antwort des Modells generieren (wir deaktivieren Filter und Leistungsbeschränkungen)
    response = chatbot(context, max_length=1000, num_return_sequences=1)[0]['generated_text']
    
    # Die Antwort extrahieren, indem der Benutzertext entfernt wird
    bot_response = response[len(context):].strip()

    # Der Verlauf wird hier nur innerhalb der Sitzung genutzt
    return bot_response, history + [(user_input, bot_response)]  # Wir geben den Verlauf zurück

# Gradio Interface: Eingabefeld und Ausgabe-Bereich
iface = gr.Interface(
    fn=chat_with_bot,
    inputs=[gr.Textbox(label="User Input"), gr.State()],
    outputs=[gr.Textbox(label="Bot Response"), gr.State()],
    live=True,
    title="Chatbot ohne Gedächtnis oder Filter",
    description="Dieser Bot gibt Antworten ohne ethische oder leistungsbedingte Filter."
)

iface.launch()