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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def data_annotations():
    @st.dialog("Add Annotations", width="large")
    def add_annotations_dialog(selected_image):
        if not os.path.exists("annotations/"):
            os.makedirs("annotations/")
        annotation_project_key = st.session_state.get('annotation_project_key', None)
        if not annotation_project_key:
            annotation_project_key = str(uuid.uuid4())
            st.session_state['annotation_project_key'] = annotation_project_key
        if not os.path.exists(f"annotations/{annotation_project_key}/"):
            os.makedirs(f"annotations/{annotation_project_key}/")
            with open(f"annotations/{annotation_project_key}/annotations.json", "w") as f:
                json.dump({}, f)
        with open(f"annotations/{annotation_project_key}/annotations.json", "r") as f:
            annotations_dict: dict = json.load(f)

        current_image_annotation = annotations_dict.get(f"images/{selected_image['file_name']}", None)
        if not current_image_annotation:
            current_image_annotation = ""

        st.image(f"images/{selected_image['file_name']}")
        annotation = st.text_area("Add Annotations", value=current_image_annotation, height=100, key="annotation_text_area")
        if st.button("Add Annotations", key="add_annotations_dialog"):
            if annotation.strip() == "" or annotation is None:
                st.warning("Please add annotations.")
            else:
                annotations_dict[f"images/{selected_image['file_name']}"] = annotation
                with open(f"annotations/{annotation_project_key}/annotations.json", "w") as f:
                    json.dump(annotations_dict, f, indent=4)
                st.toast("Annotations added successfully.", icon="🎉")
                st.rerun()

    st.title("Data Annotations")

    files = get_local_files("images/", extensions=["jpg", "jpeg", "png"], get_details=True)

    if not files:
        st.warning("No images found in the data directory.")
        return
    
    st.write(f"Total {len(files)} images found in the data directory.")

    files_df = pd.DataFrame(files)
    
    files_df['Image'] = files_df['file_name'].apply(lambda x: f"data:image/{x.split('.')[-1]};base64,{base64.b64encode(open(f'images/{x}', 'rb').read()).decode()}")

    files_df = files_df[["Image", "file_name", "file_size", "file_created"]]

    if "annotation_project_key" in st.session_state:
        annotation_project_key = st.session_state['annotation_project_key']
        if os.path.exists(f"annotations/{annotation_project_key}/annotations.json"):
            with open(f"annotations/{annotation_project_key}/annotations.json", "r") as f:
                annotations_dict: dict = json.load(f)
            files_df["Annotation"] = files_df["file_name"].apply(lambda x: annotations_dict.get(f"images/{x}", ""))

    event = st.dataframe(files_df, hide_index=True, use_container_width=True, column_config={"Image" : st.column_config.ImageColumn('Image', pinned=True)}, selection_mode="single-row", on_select='rerun', key="image_table")

    if len(event.selection['rows']):
        selected_image = files[event.selection['rows'][0]]
        add_annotations_dialog(selected_image)