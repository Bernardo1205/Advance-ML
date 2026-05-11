import torch
from torch import nn
from train_utils import train, validation
from model import CirrhosisNN


def objective(trial, train_loader, val_loader, device, epochs: int):
    h1 = trial.suggest_int("h1", 16, 128)
    h2 = trial.suggest_int("h2", 16, 128)
    h3 = trial.suggest_int("h3", 16, 128)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 1)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    scaling = trial.suggest_float("scaling", 0.9, 1)

    model = CirrhosisNN(h1=h1, h2=h2, h3=h3).to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        betas=(momentum, scaling),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        train(model, train_loader, optimizer, loss_fn, device)

    return validation(model, val_loader, loss_fn, device)