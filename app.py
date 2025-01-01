import clip
import faiss
import os
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
import time
import torch

from data_upload import data_upload_page
from data_search import data_search_page
from data_annotations import data_annotation_page
from model_finetuning import model_finetuning_page
from utils import load_clip_model, load_text_embedding_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(layout="wide", page_title="LoomRAG", page_icon="🔍")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_model()
text_embedding_model = load_text_embedding_model()
os.makedirs("annotations/", exist_ok=True)
os.makedirs("images/", exist_ok=True)

with st.sidebar:
    st.title("LoomRAG")
    page = option_menu(
        menu_title=None,
        options=["Data Upload", 'Data Search', "Data Annotation", "Model Fine-Tuning"], 
        icons=['cloud-upload', 'search', 'bi-card-checklist', 'sliders'],
        menu_icon="list", default_index=0
    )

if page == "Data Upload":
    data_upload_page.data_upload(clip_model, preprocess, text_embedding_model)
if page == "Data Search":
    data_search_page.data_search(clip_model, preprocess, text_embedding_model, device)
if page == "Data Annotation":
    data_annotation_page.data_annotations()
if page == "Model Fine-Tuning":
    model_finetuning_page.model_finetuning()
