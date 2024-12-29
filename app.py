import clip
import faiss
import os
import pandas as pd
from PIL import Image
import streamlit as st
import time
import torch

from utils import load_model, load_index
from core.vectordb import add_image_to_index, add_pdf_to_index, search_index

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_model()

if not os.path.exists("./vectorstore/faiss_index.index"):
    index = faiss.IndexFlatL2(512)
else:
    index, data = load_index()

st.title("LoomRAG")
tabs = st.tabs(["Upload Data", "Retrieve Data"])

with tabs[0]:
    upload_choice = st.selectbox(options=["Upload Image", "Upload PDF"], label="Select Upload Type")
    if upload_choice == "Upload Image":
        st.subheader("Add Image to Database")
        image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if image:
            st.image(image)
            if st.button("Add Image"):
                add_image_to_index(image, index, model, preprocess)
                st.success("Image Added to Database")
    else:
        st.subheader("Add PDF to Database")
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf:
            if st.button("Add PDF"):
                add_pdf_to_index(pdf, index, model, preprocess)
                st.success("PDF Added to Database")

with tabs[1]:
    text_input = st.text_input("Search Image Database")
    if st.button("Search", disabled=text_input.strip() == ""):
        with torch.no_grad():
            if not os.path.exists("./vectorstore/faiss_index.index"):
                st.error("No Index Found")
            else:
                data = pd.read_csv("./vectorstore/image_data.csv")
            indices = search_index(text_input, index, model, k=3)
            st.subheader("Top 3 Results")
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    image_path = data['path'].iloc[indices[0][i]]
                    if image_path.startswith("CONTENT:"):
                        st.write(image_path.replace("CONTENT: ", ""))
                        continue
                    image = Image.open(image_path)
                    image = preprocess(image).unsqueeze(0).to(device)
                    text = clip.tokenize([text_input]).to(device)
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    cosine_similarity = torch.cosine_similarity(image_features, text_features)
                    st.image(image_path)
