import clip
import clip.model
import faiss
import io
from langchain_text_splitters import CharacterTextSplitter
import os
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image
import streamlit as st
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("./vectorstore", exist_ok=True)


def add_image_to_index(image, index: faiss.IndexFlatL2, model: clip.model.CLIP, preprocess):
    image_name = image.name
    image_name = image_name.replace(" ", "_")
    os.makedirs("./images", exist_ok=True)
    os.makedirs("./vectorstore", exist_ok=True)
    with open(f"./images/{image_name}", "wb") as f:
        try:
            f.write(image.read())
        except:
            image = io.BytesIO(image.data)
            f.write(image.read())
    image = Image.open(f"./images/{image_name}")
    with torch.no_grad():
        image = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        index.add(image_features.cpu().numpy())
        if not os.path.exists("./vectorstore/image_data.csv"):
            df = pd.DataFrame([{"path": f"./images/{image_name}", "index": 0}]).reset_index(drop=True)
            df.to_csv("./vectorstore/image_data.csv", index=False)
        else:
            df = pd.read_csv("./vectorstore/image_data.csv").reset_index(drop=True)
            new_entry_df = pd.DataFrame({"path": f"./images/{image_name}", "index": len(df)}, index=[0])
            df = pd.concat([df, new_entry_df], ignore_index=True)
            df.to_csv("./vectorstore/image_data.csv", index=False)

        if not os.path.exists("./vectorstore/faiss_index.index"):
            faiss.write_index(index, './vectorstore/faiss_index.index')
        else:
            os.remove("./vectorstore/faiss_index.index")
            faiss.write_index(index, './vectorstore/faiss_index.index')
        return index


def add_pdf_to_index(pdf, index, model, preprocess):
    pdf_name = pdf.name
    pdf_name = pdf_name.replace(" ", "_")
    pdf_reader = PdfReader(pdf)
    pdf_pages_data = []
    pdf_texts = []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=150,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )
    progress_bar = st.progress(0)
    for page_num, page in enumerate(pdf_reader.pages):
        try:
            page_images = page.images
        except:
            page_images = []
            st.error("Some images in the PDF are not readable. Please try another PDF.")
        for image in page_images:
            image.name = f"{time.time()}.png"
            add_image_to_index(image, index, model, preprocess)
            pdf_pages_data.append({f"page_number": page_num, "content": image, "type": "image"})

        page_text = page.extract_text()
        pdf_texts.append(page_text)
        if page_text != "" or page_text.strip() != "":
            chunks = text_splitter.split_text(page_text)
            for chunk in chunks:
                text = clip.tokenize([chunk]).to(device)
                text_features = model.encode_text(text)
                index.add(text_features.detach().numpy())
                df = pd.read_csv("./vectorstore/image_data.csv").reset_index(drop=True)
                new_entry_df = pd.DataFrame({"path": f"CONTENT: {chunk}", "index": len(df)}, index=[0])
                df = pd.concat([df, new_entry_df], ignore_index=True)
                df.to_csv("./vectorstore/image_data.csv", index=False)
                pdf_pages_data.append({f"page_number": page_num, "content": chunk, "type": "text"})

            if not os.path.exists("./vectorstore/faiss_index.index"):
                faiss.write_index(index, './vectorstore/faiss_index.index')
            else:
                os.remove("./vectorstore/faiss_index.index")
                faiss.write_index(index, './vectorstore/faiss_index.index')
        percent_complete = ((page_num) / len(pdf_reader.pages))
        progress_bar.progress(percent_complete, f"Processing Page {page_num + 1}/{len(pdf_reader.pages)}")
    return pdf_pages_data



def search_index(text_input, index, model, k=3):
    with torch.no_grad():
        text = clip.tokenize([text_input]).to(device)
        text_features = model.encode_text(text)
        distances, indices = index.search(text_features.cpu().numpy(), k)
        return indices
