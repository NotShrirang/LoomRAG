import clip
from datetime import datetime
import faiss
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
import whisper

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

@st.cache_resource
def load_text_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("small")
    return model

def load_image_index():
    index = faiss.read_index('./vectorstore/image_index.index')
    data = pd.read_csv("./vectorstore/image_data.csv")
    return index, data

def load_text_index():
    index = faiss.read_index('./vectorstore/text_index.index')
    data = pd.read_csv("./vectorstore/text_data.csv")
    return index, data

def load_audio_index():
    index = faiss.read_index('./vectorstore/audio_index.index')
    data = pd.read_csv("./vectorstore/audio_data.csv")
    return index, data

def cosine_similarity(a, b):
    return torch.cosine_similarity(a, b)


def get_local_files(directory: str, extensions: list = None, get_details: bool = False):
    files = os.listdir(directory)
    if not extensions:
        if get_details:
            return [{
                "file_name": file,
                "file_size": os.path.getsize(os.path.join(directory, file)),
                "file_created": datetime.fromtimestamp(os.path.getctime(os.path.join(directory, file)))
            } for file in files]
        else:
            return files
    else:
        if get_details:
            filtered_files = []
            for file in files:
                file_extension = file.split(".")[-1]
                if file_extension in extensions:
                    filtered_files.append({
                        "file_name": file,
                        "file_size": os.path.getsize(os.path.join(directory, file)),
                        "file_created": datetime.fromtimestamp(os.path.getctime(os.path.join(directory, file)))
                    })
            return filtered_files
        else:
            return [file for file in files if file.endswith(extensions(extensions))]
