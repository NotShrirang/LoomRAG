import base64
import clip
import io
import json
import os
import pandas as pd
from PIL import Image
import streamlit as st
import sys
import torch
import uuid

from utils import get_local_files
from model_finetuning.components import selection_component, model_training_component

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def model_finetuning():
    page = st.session_state.get("model_finetuning_page", 0)

    pages = {
        0: "Preference Selection",
        1: "Model Fine-Tuning",
        2: "Model Evaluation"
    }
    st.title(pages[page])
    
    if not torch.cuda.is_available():
        st.warning("No GPUs detected. Model training should be done on a GPU machine or on remote instance.")

    if page == 0:
        selection_component.preference_selection()
    elif page == 1:
        model_training_component.model_training()
    elif page == 2:
        st.write("Model Evaluation")

    col1, col2 = st.columns(2, gap="large")
    if col1.button("⬅️ Back", disabled=page == 0, use_container_width=True):
        st.session_state["model_finetuning_page"] = page - 1
        st.rerun()
    if col2.button("Next ➡️", disabled=page == 2, use_container_width=True):
        st.session_state["model_finetuning_page"] = page + 1
        st.rerun()

    
