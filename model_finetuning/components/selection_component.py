import clip
import json
import pandas as pd
import streamlit as st
import torch
import time

from utils import get_local_files

def preference_selection():
    annotation_projects = get_local_files("annotations/", get_details=True)
    if not annotation_projects:
        st.warning("No annotated data found.")
        return

    annotation_projects_df = pd.DataFrame(annotation_projects)
    annotation_projects_df['file_created'] = annotation_projects_df['file_created'].dt.strftime("%Y-%m-%d %H:%M:%S")
    annotation_projects_df['display_text'] = annotation_projects_df.apply(lambda x: f"ID: {x['file_name']} | Time Created: ({x['file_created']})", axis=1)

    annotation_project = st.selectbox("Select Annotation Project", options=annotation_projects_df['display_text'].tolist())
    annotation_project = annotation_projects_df[annotation_projects_df['display_text'] == annotation_project].iloc[0]
    with open(f"annotations/{annotation_project['file_name']}/annotations.json", "r") as f:
        annotations_dict: dict = json.load(f)

    annotations_df = pd.DataFrame(annotations_dict.items(), columns=['image_path', 'annotation'])
    annotations_df['image_path'] = annotations_df['image_path'].apply(lambda x: x.split('/')[-1])
    cols = st.columns(5)
    for i, row in annotations_df.head(4).iterrows():
        with cols[i]:
            st.image(f"images/{row['image_path']}", caption=row['annotation'])
    if len(annotations_df) > 4:
        with cols[4]:
            st.info(f"and more {len(annotations_df) - 4} images...")

    save_preference = st.button("Save Preferences")
    if save_preference:
        st.session_state['selected_dataset'] = annotation_project['file_name']
        st.success("Preferences Saved Successfully.")
