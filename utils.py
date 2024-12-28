import clip
import faiss
import pandas as pd
import streamlit as st
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

@st.cache_resource
def load_index():
    index = faiss.read_index('./vectorstore/faiss_index.index')
    data = pd.read_csv("./vectorstore/image_data.csv")
    return index, data
