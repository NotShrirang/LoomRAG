import clip
import faiss
import os
import pandas as pd
from PIL import Image
import streamlit as st
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("./vectorstore", exist_ok=True)

def add_image_to_index(image, index, model, preprocess):
    image_name = image.name
    image_name = image_name.replace(" ", "_")
    with open(f"./images/{image_name}", "wb") as f:
        f.write(image.read())
    image = Image.open(f"./images/{image_name}")
    with torch.no_grad():
        image = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        index.add(image_features.cpu().numpy())
        if not os.path.exists("./vectorstore/image_data.csv"):
            df = pd.DataFrame([{"path": f"./images/{image}", "index": 0}]).reset_index(drop=True)
            df.to_csv("./vectorstore/image_data.csv", index=False)
        else:
            df = pd.read_csv("./vectorstore/image_data.csv").reset_index(drop=True)
            new_entry_df = pd.DataFrame({"path": f"./images/{image_name}", "index": len(df)}, index=[0])
            df = pd.concat([df, new_entry_df], ignore_index=True)
            df.to_csv("./vectorstore/image_data.csv", index=False)
        return index

def search_index(text_input, index, model, k=3):
    with torch.no_grad():
        text = clip.tokenize([text_input]).to(device)
        text_features = model.encode_text(text)
        distances, indices = index.search(text_features.cpu().numpy(), k)
        return indices
