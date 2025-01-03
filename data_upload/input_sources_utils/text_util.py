import bs4
import os
from langchain_text_splitters import CharacterTextSplitter
import requests
import streamlit as st
import sys
from vectordb import add_image_to_index, add_pdf_to_index, update_vectordb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def process_text(text: str, text_embedding_model):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    text_embeddings = text_embedding_model.encode(chunks)
    for chunk, embedding in zip(chunks, text_embeddings):
        index = update_vectordb(index_path="text_index.index", embedding=embedding, text_content=chunk)
    return index
