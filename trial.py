import torch
import clip
from PIL import Image
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("./images/temp.png")).unsqueeze(0).to(device)
text = clip.tokenize(["pokemon", "a boy", "a girl holding pokeball"]).to(device)

index = faiss.IndexFlatL2(512)

with torch.no_grad():
    image_features = model.encode_image(image)
    print(image_features.cpu().numpy().shape)
    index.add(image_features.cpu().numpy())
    text_features = model.encode_text(text)
    print(text_features.cpu().numpy().shape)
    index.add(text_features.cpu().numpy())
    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy().round(3)
    sample_text = "Image of pokemon charactor"
    text = clip.tokenize([sample_text]).to(device)
    text_features = model.encode_text(text)
    D, I = index.search(text_features.cpu().numpy(), 3)

    # query_embedding = image_features.cpu().numpy()
    # D, I = index.search(query_embedding, 3)
    # index
    print("I:", I)
    print("D:", D)

# print("Label probs:", probs)