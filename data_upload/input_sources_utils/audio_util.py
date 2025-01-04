import os
import requests
import streamlit as st
import sys
import whisper

from vectordb import add_image_to_index, add_pdf_to_index, add_audio_to_index

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def upload_audio(whisper_model, text_embedding_model):
    st.title("Upload Audio")
    recorded_audio = st.audio_input("Record Audio")
    st.write("---")
    uploaded_audios = st.file_uploader("Upload Audio", type=["mp3", "wav"], accept_multiple_files=True)
    if recorded_audio:
        st.audio(recorded_audio)
        if st.button("Add Audio"):
            add_audio_to_index(recorded_audio, whisper_model, text_embedding_model)
            st.success("Audio Added to Database")
    if uploaded_audios:
        for audio in uploaded_audios:
            st.audio(audio)
        if st.button("Add Audio"):
            progress_bar = st.progress(0, f"Adding Audio... | 0/{len(uploaded_audios)}")
            for count, audio in enumerate(uploaded_audios):
                add_audio_to_index(audio, whisper_model, text_embedding_model)
                progress_bar.progress((count + 1) / len(uploaded_audios), f"Adding Audio... | {count + 1}/{len(uploaded_audios)}")
            st.success("Audio Added to Database")
