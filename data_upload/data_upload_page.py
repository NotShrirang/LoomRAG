import os
import streamlit as st
import sys
from vectordb import add_image_to_index, add_pdf_to_index

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def data_upload(clip_model, preprocess, text_embedding_model):
    st.title("Data Upload")
    upload_choice = st.selectbox(options=["Upload Image", "Upload PDF"], label="Select Upload Type")
    if upload_choice == "Upload Image":
        st.subheader("Add Image to Database")
        images = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if images:
            cols = st.columns(5, vertical_alignment="center")
            for count, image in enumerate(images[:4]):
                with cols[count]:
                    st.image(image)
            with cols[4]:
                if len(images) > 5:
                    st.info(f"and more {len(images) - 5} images...")
            st.info(f"Total {len(images)} files selected.")
            if st.button("Add Images"):
                progress_bar = st.progress(0)
                for image in images:
                    add_image_to_index(image, clip_model, preprocess)
                    progress_bar.progress((images.index(image) + 1) / len(images), f"{images.index(image) + 1}/{len(images)}")
                st.success("Images Added to Database")
    else:
        st.subheader("Add PDF to Database")
        st.warning("Please note that the images in the PDF will also be extracted and added to the database.")
        pdfs = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
        if pdfs:
            st.info(f"Total {len(pdfs)} files selected.")
            if st.button("Add PDF"):
                for pdf in pdfs:
                    add_pdf_to_index(
                        pdf=pdf,
                        clip_model=clip_model,
                        preprocess=preprocess,
                        text_embedding_model=text_embedding_model,
                    )
                st.success("PDF Added to Database")