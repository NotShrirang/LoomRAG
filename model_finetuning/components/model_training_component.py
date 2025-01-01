import clip
import clip.model
from datasets import Dataset
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import streamlit as st
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import os


def model_training():
    dataset_path = st.session_state.get("selected_dataset", None)
    if not dataset_path:
        st.error("Please select a dataset to proceed.")
        return
    
    if not os.path.exists(f"annotations/{dataset_path}/annotations.json"):
        st.error("No annotations found for the selected dataset.")
        return
    
    with open(f"annotations/{dataset_path}/annotations.json", "r") as f:
        annotations_dict = json.load(f)

    annotations_df = pd.DataFrame(annotations_dict.items(), columns=['image_path', 'annotation'])
    annotations_df.columns = ['file_name', 'text']
    st.subheader("Data Preview")
    st.dataframe(annotations_df.head(), use_container_width=True)

    test_size = st.selectbox("Select Test Size", options=[0.1, 0.2, 0.3, 0.4, 0.5], index=1)
    train_df, val_df = train_test_split(annotations_df, test_size=test_size, random_state=42)
    st.write(f"Train Size: {len(train_df)} | Validation Size: {len(val_df)}")
    col1, col2 = st.columns(2)
    with col1:
        optimizer = st.selectbox("Select Optimizer", options=optim.__all__, index=3)
        optimizer = getattr(optim, optimizer)
    with col2:
        batch_size_options = [2, 4, 8, 16, 32, 64, 128]
        ideal_batch_size = int(np.sqrt(len(train_df)))
        if ideal_batch_size in batch_size_options:
            ideal_batch_size_index = batch_size_options.index(ideal_batch_size)
        else:
            for batch_size in batch_size_options:
                if batch_size > ideal_batch_size:
                    ideal_batch_size_index = batch_size_options.index(batch_size) - 1
                    break
        batch_size = st.selectbox("Select Batch Size", options=[2, 4, 8, 16, 32, 64, 128], index=ideal_batch_size_index)
    
    col1, col2 = st.columns(2)
    with col1:
        weight_decay = st.number_input("Weight Decay", value=0.3, format="%.5f")
    with col2:
        learning_rate = st.number_input("Learning Rate", value=1e-3, format="%.5f")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if st.button("Train", key="train_button", use_container_width=True, type="primary"):
        def convert_models_to_fp32(model):
            for p in model.parameters():
                p.data = p.data.float()
                p.grad.data = p.grad.data.float()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with st.spinner("Loading Model..."):
            model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
            clip.model.convert_weights(model)

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        optimizer = optimizer(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=weight_decay)

        def collate_fn(batch):
            images = []
            texts = []
            for entry in batch:
                img = entry['file_name']
                text = entry['text']
                images.append(img)
                texts.append(text)
            images = [preprocess(Image.open(img_path)) for img_path in images]
            images = torch.stack(images)
            return images, list(texts)
        
        train_df['file_name'] = train_df['file_name'].str.strip()
        val_df['file_name'] = val_df['file_name'].str.strip()

        dataset = Dataset.from_pandas(train_df)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_dataset = Dataset.from_pandas(val_df)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        def calculate_val_loss(model):
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch_idx, (images, texts) in enumerate(val_dataloader):
                    texts = clip.tokenize(texts).to(device)

                    images = images.to(device)
                    texts = texts.to(device)

                    logits_per_image, logits_per_text = model(images, texts)

                    ground_truth = torch.arange(len(images)).to(device)
                    image_loss = loss_img(logits_per_image, ground_truth)
                    text_loss = loss_txt(logits_per_text, ground_truth)

                    total_loss += (image_loss + text_loss) / 2

            model.train()
            return total_loss / len(val_dataloader)

        step = 0
        progress_bar = st.progress(0, text=f"Model Training in progress... \nStep: {step}/{len(dataloader)} | {0 / len(dataloader)}% Completed | Loss: 0.0")
        for batch_idx, (images, texts) in enumerate(dataloader):
            optimizer.zero_grad()

            texts = clip.tokenize(texts).to(device)

            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images)).to(device)
            image_loss = loss_img(logits_per_image, ground_truth)
            text_loss = loss_txt(logits_per_text, ground_truth)
            total_loss = (image_loss + text_loss) / 2
            total_loss.backward()

            if step % 20 == 0:
                print("\nStep : ", step)
                print("Total Loss : ", total_loss.item())
                val_loss = calculate_val_loss(model)
                print("\nValidation Loss : ", val_loss.item())
                print("\n")

            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
            step += 1
            progress_bar.progress((batch_idx + 1) / len(dataloader), f"Model Training in progress... \nStep: {step}/{len(dataloader)} | {round((batch_idx + 1) / len(dataloader) * 100)}% Completed | Loss: {val_loss.item():.4f}")

        st.toast("Training Completed!", icon="🎉")

        with st.spinner("Saving Model..."):
            finetuned_model = model.eval()
            torch.save(finetuned_model.state_dict(), f"annotations/{dataset_path}/finetuned_model.pt")

        st.success("Model Saved Successfully!")
