import torch
import clip
from PIL import Image
import faiss
import os
import streamlit as st
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

index: faiss.IndexFlatL2 = faiss.read_index('./vectorstore/faiss_index.index')

data = pd.read_csv("./vectorstore/image_data.csv")

st.title("MultiModal RAG")

text_input = st.text_input("Search Image Database")

if st.button("Search"):
    with torch.no_grad():
        text = clip.tokenize([text_input]).to(device)
        text_features = model.encode_text(text)
        D, I = index.search(text_features.cpu().numpy(), 3)
        cols = st.columns(3)
        for i in range(3):
            with cols[i]:
                st.write(data['path'].iloc[I[0][i]])
                st.image(data['path'].iloc[I[0][i]])