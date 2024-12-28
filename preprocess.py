import torch
import clip
from PIL import Image
import faiss
import os
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

index = faiss.IndexFlatL2(512)

os.makedirs("./vectorstore", exist_ok=True)

with torch.no_grad():
    image_list = []
    for idx, file in enumerate(os.listdir("./images")):
        print(file)
        image = preprocess(Image.open(f"./images/{file}")).unsqueeze(0).to(device)
        image_data = {'path': f"./images/{file}", "index": idx}
        image_features = model.encode_image(image)
        index.add(image_features.cpu().numpy())
        image_list.append(image_data)

df = pd.DataFrame(image_list, columns=['path', 'index'])
df.to_csv("./vectorstore/image_data.csv")
faiss.write_index(index, './vectorstore/faiss_index.index')