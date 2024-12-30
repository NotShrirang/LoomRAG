import clip
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

@st.cache_resource
def load_text_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def load_image_index():
    index = faiss.read_index('./vectorstore/image_index.index')
    data = pd.read_csv("./vectorstore/image_data.csv")
    return index, data

def load_text_index():
    index = faiss.read_index('./vectorstore/text_index.index')
    data = pd.read_csv("./vectorstore/text_data.csv")
    return index, data

def cosine_similarity(a, b):
    return torch.cosine_similarity(a, b)