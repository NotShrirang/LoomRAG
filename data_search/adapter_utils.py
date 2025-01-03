import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, img_dim, txt_dim, embed_dim=1024, num_heads=8):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(img_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, txt_dim)
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=txt_dim, num_heads=num_heads)

    def forward(self, img_emb):
        img_emb = self.adapter(img_emb).unsqueeze(0)
        attn_output, _= self.self_attention(img_emb, img_emb, img_emb)
        return attn_output.squeeze(0)


def get_adapter_model(in_shape, out_shape):
    model = Adapter(in_shape, out_shape)
    return model


def load_adapter_model():
    model = get_adapter_model(512, 384)
    model.load_state_dict(torch.load("./weights/adapter_model_with_attention.pt", map_location=torch.device('cpu')))
    return model
