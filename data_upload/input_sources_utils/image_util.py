import os
import requests
import streamlit as st
import sys
from vectordb import add_image_to_index, add_pdf_to_index

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def image_from_url(clip_model, preprocess):
    st.title("Image from URL")
    url = st.text_input("Enter Image URL")
    correct_url = False
    if url:
        try:
            st.image(url)
            correct_url = True
        except:
            st.error("Invalid URL")
            correct_url = False
        if correct_url:
            if st.button("Add Image"):
                response = requests.get(url)
                if response.status_code == 200:
                    add_image_to_index(response.content, clip_model, preprocess)
                    st.success("Image Added to Database")
                else:
                    st.error("Invalid URL")

def upload_image(clip_model, preprocess):
    st.subheader("Add Image to Database")
    images = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if images:
        cols = st.columns(5, vertical_alignment="center")
        for count, image in enumerate(images[:4]):
            with cols[count]:
                st.image(image)
        with cols[4]:
            if len(images) > 5:
                st.info(f"and more {len(images) - 5} images...")
        st.info(f"Total {len(images)} files selected.")
        if st.button("Add Images"):
            progress_bar = st.progress(0)
            for image in images:
                add_image_to_index(image, clip_model, preprocess)
                progress_bar.progress((images.index(image) + 1) / len(images), f"{images.index(image) + 1}/{len(images)}")
            st.success("Images Added to Database")