import clip
import clip.model
import faiss
import io
from langchain_text_splitters import CharacterTextSplitter
import os
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("./vectorstore", exist_ok=True)

def update_vectordb(index_path: str, embedding: torch.Tensor, image_path: str = None, text_content: str = None):
    if not image_path and not text_content:
        raise ValueError("Either image_path or text_content must be provided.")
    if not os.path.exists(f"./vectorstore/{index_path}"):
        if image_path:
            index = faiss.IndexFlatL2(512)
        else:
            index = faiss.IndexFlatL2(384)
    else:
        index = faiss.read_index(f"./vectorstore/{index_path}")
    try:
        index.add(embedding.cpu().numpy())
    except:
        if len(embedding.shape) == 1:
            embedding = torch.Tensor([embedding])
        index.add(embedding)
    faiss.write_index(index, f'./vectorstore/{index_path}')
    if image_path:
        if not os.path.exists("./vectorstore/image_data.csv"):
            df = pd.DataFrame([{"path": image_path, "index": 0}]).reset_index(drop=True)
            df.to_csv("./vectorstore/image_data.csv", index=False)
        else:
            df = pd.read_csv("./vectorstore/image_data.csv").reset_index(drop=True)
            new_entry_df = pd.DataFrame({"path": image_path, "index": len(df)}, index=[0])
            df = pd.concat([df, new_entry_df], ignore_index=True)
            df.to_csv("./vectorstore/image_data.csv", index=False)
    elif text_content:
        if not os.path.exists("./vectorstore/text_data.csv"):
            df = pd.DataFrame([{"content": text_content, "index": 0}]).reset_index(drop=True)
            df.to_csv("./vectorstore/text_data.csv", index=False)
        else:
            df = pd.read_csv("./vectorstore/text_data.csv").reset_index(drop=True)
            new_entry_df = pd.DataFrame({"content": text_content, "index": len(df)}, index=[0])
            df = pd.concat([df, new_entry_df], ignore_index=True)
            df.to_csv("./vectorstore/text_data.csv", index=False)
    else:
        raise ValueError("Either image_path or text_content must be provided.")
    return index


def add_image_to_index(image, model: clip.model.CLIP, preprocess):
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
        index = update_vectordb(index_path="image_index.index", embedding=image_features, image_path=f"./images/{image_name}")
        return index


def add_pdf_to_index(pdf, clip_model: clip.model.CLIP, preprocess, text_embedding_model: SentenceTransformer):
    if not os.path.exists("./vectorstore/"):
        os.makedirs("./vectorstore")
    pdf_name = pdf.name
    pdf_name = pdf_name.replace(" ", "_")
    pdf_reader = PdfReader(pdf)
    pdf_pages_data = []
    pdf_texts = []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
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
            add_image_to_index(image, clip_model, preprocess)
            pdf_pages_data.append({f"page_number": page_num, "content": image, "type": "image"})

        page_text = page.extract_text()
        pdf_texts.append(page_text)
        if page_text != "" or page_text.strip() != "":
            chunks = text_splitter.split_text(page_text)
            text_embeddings: torch.Tensor = text_embedding_model.encode(chunks)
            for i, chunk in enumerate(chunks):
                update_vectordb(index_path="text_index.index", embedding=text_embeddings[i], text_content=chunk)
                pdf_pages_data.append({f"page_number": page_num, "content": chunk, "type": "text"})
        percent_complete = ((page_num + 1) / len(pdf_reader.pages))
        progress_bar.progress(percent_complete, f"Processing Page {page_num + 1}/{len(pdf_reader.pages)}")
    return pdf_pages_data


def search_image_index(text_input: str, index: faiss.IndexFlatL2, clip_model: clip.model.CLIP, k: int = 3):
    with torch.no_grad():
        text = clip.tokenize([text_input]).to(device)
        text_features = clip_model.encode_text(text)
        distances, indices = index.search(text_features.cpu().numpy(), k)
        return indices

def search_text_index(text_input: str, index: faiss.IndexFlatL2, text_embedding_model: SentenceTransformer, k: int = 3):
    text_embeddings = text_embedding_model.encode([text_input])
    distances, indices = index.search(text_embeddings, k)
    return indices