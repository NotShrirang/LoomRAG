import os
import streamlit as st
import sys
from vectordb import add_image_to_index, add_pdf_to_index

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def upload_pdf(clip_model, preprocess, text_embedding_model):
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