import clip
import os
import pandas as pd
from PIL import Image
import streamlit as st
import sys
import torch
from vectordb import search_image_index, search_text_index, search_image_index_with_image, search_text_index_with_image
from utils import load_image_index, load_text_index, load_audio_index, get_local_files
from data_search import adapter_utils

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def data_search(clip_model, preprocess, text_embedding_model, whisper_model, device):

    @st.cache_resource
    def load_finetuned_model(file_name):
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.load_state_dict(torch.load(f"annotations/{file_name}/finetuned_model.pt", weights_only=True))
        return model, preprocess
    
    @st.cache_resource
    def load_adapter():
        adapter = adapter_utils.load_adapter_model()
        return adapter

    st.title("Data Search")

    images = os.listdir("images/")
    if images == []:
        st.warning("No Images Found! Please upload images to the database.")
        return

    annotation_projects = get_local_files("annotations/", get_details=True)

    if annotation_projects or st.session_state.get('selected_annotation_project', None) is not None:
        annotation_projects_with_model = []
        for annotation_project in annotation_projects:
            if os.path.exists(f"annotations/{annotation_project['file_name']}/finetuned_model.pt"):
                annotation_projects_with_model.append(annotation_project)

        if annotation_projects_with_model or st.session_state.get('selected_annotation_project', None) is not None:
            if st.button("Use Default Model"):
                st.session_state.pop('selected_annotation_project', None)
            annotation_projects_df = pd.DataFrame(annotation_projects_with_model)
            annotation_projects_df['file_created'] = annotation_projects_df['file_created'].dt.strftime("%Y-%m-%d %H:%M:%S")
            annotation_projects_df['display_text'] = annotation_projects_df.apply(lambda x: f"ID: {x['file_name']} | Time Created: ({x['file_created']})", axis=1)

            annotation_project = st.selectbox("Select Annotation Project", options=annotation_projects_df['display_text'].tolist())
            annotation_project = annotation_projects_df[annotation_projects_df['display_text'] == annotation_project].iloc[0]
            if st.button("Use Selected Fine-Tuned Model") or st.session_state.get('selected_annotation_project', None) is not None:
                with st.spinner("Loading Fine-Tuned Model..."):
                    st.session_state['selected_annotation_project'] = annotation_project
                    clip_model, preprocess = load_finetuned_model(annotation_project['file_name'])
                st.info(f"Using Fine-Tuned Model from {annotation_project['file_name']}")
            else:
                st.info("Using Default Model")

    adapter = load_adapter()
    adapter.to(device)

    text_input = st.text_input("Search Database")
    image_input = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if st.button("Search", disabled=text_input.strip() == "" and image_input is None):
        if os.path.exists("./vectorstore/image_index.index"):
            image_index, image_data = load_image_index()
        if os.path.exists("./vectorstore/text_index.index"):
            text_index, text_data = load_text_index()
        if os.path.exists("./vectorstore/audio_index.index"):
            audio_index, audio_data = load_audio_index()
        with torch.no_grad():
            if not os.path.exists("./vectorstore/image_data.csv"):
                st.warning("No Image Index Found. So not searching for images.")
                image_index = None
            if not os.path.exists("./vectorstore/text_data.csv"):
                st.warning("No Text Index Found. So not searching for text.")
                text_index = None
            if not os.path.exists("./vectorstore/audio_data.csv"):
                st.warning("No Audio Index Found. So not searching for audio.")
            if image_input:
                image = Image.open(image_input)
                image = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image)
                    adapted_text_embeddings = adapter(image_features)
                    if image_index is not None:
                        image_indices = search_image_index_with_image(image_features, image_index, clip_model, k=3)
                    if text_index is not None:
                        text_indices = search_text_index_with_image(adapted_text_embeddings, text_index, text_embedding_model, k=3)
                    if audio_index is not None:
                        audio_indices = search_text_index_with_image(adapted_text_embeddings, audio_index, text_embedding_model, k=3)
            else:
                if image_index is not None:
                    image_indices = search_image_index(text_input, image_index, clip_model, k=3)
                if text_index is not None:
                    text_indices = search_text_index(text_input, text_index, text_embedding_model, k=3)
                if audio_index is not None:
                    audio_indices = search_text_index(text_input, audio_index, text_embedding_model, k=3)
            if not image_index and not text_index and not audio_index:
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
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    if audio_index:
                        audio_path = audio_data['path'].iloc[audio_indices[0][i]]
                        st.audio(audio_path)