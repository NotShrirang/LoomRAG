import os
import streamlit as st
import sys

from data_upload.input_sources_utils import image_util, pdf_util, website_util, audio_util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def data_upload(clip_model, preprocess, text_embedding_model, whisper_model):
    st.title("Data Upload")
    st.warning("Please note that this is a public application. Make sure you are not uploading any sensitive data.")
    upload_choice = st.selectbox(options=["Upload Image", "Add Image from URL / Link", "Upload PDF", "Website Link", "Audio Recording"], label="Select Upload Type")
    if upload_choice == "Upload Image":
        image_util.upload_image(clip_model, preprocess)
    elif upload_choice == "Add Image from URL / Link":
        image_util.image_from_url(clip_model, preprocess)
    elif upload_choice == "Upload PDF":
        pdf_util.upload_pdf(clip_model, preprocess, text_embedding_model)
    elif upload_choice == "Website Link":
        website_util.data_from_website(clip_model, preprocess, text_embedding_model)
    elif upload_choice == "Audio Recording":
        audio_util.upload_audio(whisper_model, text_embedding_model)
