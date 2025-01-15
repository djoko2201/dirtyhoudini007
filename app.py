import streamlit as st
from transformers import pipeline

# Lade das Modell von Hugging Face (Beispielmodell hier verwenden, ersetze mit deinem Modell)
chatbot = pipeline("conversational", model="cognitivecomputations/dolphin-2.9.2-qwen2-72b")

# Definiere den Titel und die Beschreibung der Anwendung
st.title("Chatbot Anwendung")
st.write("Dies ist ein interaktiver Chatbot. Du kannst Nachrichten eingeben und der Chatbot wird antworten.")

# Datei-Upload-Komponente
uploaded_file = st.file_uploader("Lade eine Datei hoch", type=["txt", "pdf", "csv"])

# Wenn eine Datei hochgeladen wurde, wird sie verarbeitet
if uploaded_file is not None:
    # Dateiinhalt anzeigen
    st.write(f"Dateiname: {uploaded_file.name}")
    st.write(f"Dateigröße: {uploaded_file.size} Bytes")
    file_content = uploaded_file.read().decode("utf-8")  # Beispiel für Textdateien
    st.text_area("Dateiinhalt", value=file_content, height=300)

# Eingabefeld für den Benutzer
user_input = st.text_input("Gib eine Nachricht ein:")

# Wenn eine Nachricht eingegeben wurde, den Chatbot aufrufen und antworten
if user_input:
    response = chatbot(user_input)
    st.write(f"Chatbot Antwort: {response[0]['generated_text']}")