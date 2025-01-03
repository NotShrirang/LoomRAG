import torch
import torch.nn as nn


def get_adapter_model(in_shape, out_shape):
    model = nn.Sequential(
        nn.Linear(in_shape, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, out_shape)
    )
    return model


def load_adapter_model():
    model = get_adapter_model(512, 384)
    model.load_state_dict(torch.load("./weights/adapter_model.pt", map_location=torch.device('cpu')))
    return model
