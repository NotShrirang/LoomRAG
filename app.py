import clip
import faiss
import os
import pandas as pd
from PIL import Image
import streamlit as st
import time
import torch

from utils import load_clip_model, load_text_embedding_model, load_image_index, load_text_index
from core.vectordb import add_image_to_index, add_pdf_to_index, search_image_index, search_text_index

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_model()
text_embedding_model = load_text_embedding_model()

st.title("LoomRAG")
tabs = st.tabs(["Upload Data", "Retrieve Data"])

with tabs[0]:
    upload_choice = st.selectbox(options=["Upload Image", "Upload PDF"], label="Select Upload Type")
    if upload_choice == "Upload Image":
        st.subheader("Add Image to Database")
        images = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if images:
            cols = st.columns(5, vertical_alignment="center")
            for count, image in enumerate(images[:4]):
                with cols[count]:
                    st.image(image)
            with cols[4]:
                st.info(f"and more {len(images) - 5} images...")
            st.info(f"Total {len(images)} files selected.")
            if st.button("Add Images"):
                progress_bar = st.progress(0)
                for image in images:
                    add_image_to_index(image, clip_model, preprocess)
                    progress_bar.progress((images.index(image) + 1) / len(images), f"{images.index(image) + 1}/{len(images)}")
                st.success("Images Added to Database")
    else:
        st.subheader("Add PDF to Database")
        pdfs = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
        if pdfs:
            st.info(f"Total {len(pdfs)} files selected.")
            if st.button("Add PDF"):
                for pdf in pdfs:
                    add_pdf_to_index(
                        pdf=pdf,
                        clip_model=clip_model,
                        preprocess=preprocess,
                        text_embedding_model=text_embedding_model,
                    )
                st.success("PDF Added to Database")

with tabs[1]:
    text_input = st.text_input("Search Database")
    if st.button("Search", disabled=text_input.strip() == ""):
        if os.path.exists("./vectorstore/image_index.index"):
            image_index, image_data = load_image_index()
        if os.path.exists("./vectorstore/text_index.index"):
            text_index, text_data = load_text_index()
        with torch.no_grad():
            if not os.path.exists("./vectorstore/image_data.csv"):
                st.warning("No Image Index Found. So not searching for images.")
                image_index = None
            if not os.path.exists("./vectorstore/text_data.csv"):
                st.warning("No Text Index Found. So not searching for text.")
                text_index = None
            if image_index is not None:
                image_indices = search_image_index(text_input, image_index, clip_model, k=3)
            if text_index is not None:
                text_indices = search_text_index(text_input, text_index, text_embedding_model, k=3)
            if not image_index and not text_index:
                st.error("No Data Found! Please add data to the database.")
            st.subheader("Top 3 Results")
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    if image_index:
                        image_path = image_data['path'].iloc[image_indices[0][i]]
                        image = Image.open(image_path)
                        image = preprocess(image).unsqueeze(0).to(device)
                        text = clip.tokenize([text_input]).to(device)
                        image_features = clip_model.encode_image(image)
                        text_features = clip_model.encode_text(text)
                        cosine_similarity = torch.cosine_similarity(image_features, text_features)
                        st.write(f"Similarity: {cosine_similarity.item() * 100:.2f}%")
                        st.image(image_path)
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    if text_index:
                        text_content = text_data['content'].iloc[text_indices[0][i]]
                        st.write(text_content)
