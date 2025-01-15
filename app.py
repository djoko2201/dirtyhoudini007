import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Laden des GPT-2 Modells und Tokenizers
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Anpassung der Antwortgenerierung mit erweiterten Sampling-Techniken und maximaler Leistung
def generate_response(input_text):
    # Tokenisiere den Eingabetext
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # Konfiguration der Antwortgenerierung (maximiert für beste Leistung und Präzision)
    outputs = model.generate(
        inputs,
        max_length=150,  # Maximale Textlänge für präzisere Antworten
        min_length=50,   # Minimale Textlänge für vollständige Antworten
        temperature=0.0,  # Bestimmt die Deterministik der Antwort (maximal fokussiert)
        top_p=0.95,      # Top-p Sampling, um die Antwortqualität zu maximieren (höhere Wahrscheinlichkeit für diverse Antworten)
        top_k=50,        # Begrenzung der möglichen nächsten Tokens (für fokussierte und präzise Antworten)
        no_repeat_ngram_size=3,  # Verhindert Wiederholungen von N-Grammen (Kohärenz)
        repetition_penalty=2.0,  # Bestraft Wiederholungen (für präzisere und weniger redundante Antworten)
        length_penalty=1.0,      # Optimierung der Textlänge
        num_return_sequences=1,  # Eine einzige Antwort zurückgeben (vermeidet unnötige Ergebnisse)
        eos_token_id=tokenizer.eos_token_id,  # Modellsoll korrekt stoppen
        bad_words_ids=[tokenizer.encode(t)[0] for t in ['Hass', 'Gewalt']],  # Verhindert toxische Antworten
        pad_token_id=tokenizer.eos_token_id,  # Sicherstellen, dass das Modell korrekt stoppt
        no_repeat_ngram_size=3,  # Verhindert Wiederholungen
        do_sample=False          # Keine Zufallselemente, deterministische Antwort
    )
    
    # Dekodiere die Antwort
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio Interface mit maximierter Antwortqualität
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    live=True,
    allow_flagging="never",  # Flagging deaktivieren, keine Benutzer-Feedback-Aufforderung
    title="Optimierter GPT-2 Chatbot",
    description="Ein hochoptimierter GPT-2-Chatbot für präzise und fokussierte Antworten.",
    theme="dark",  # Dunkles Design für bessere Lesbarkeit
    examples=[
        ["Wie funktioniert maschinelles Lernen?"],
        ["Was ist der Unterschied zwischen KI und maschinellem Lernen?"],
        ["Erkläre den Unterschied zwischen Supervised und Unsupervised Learning."]
    ]
)

iface.launch()