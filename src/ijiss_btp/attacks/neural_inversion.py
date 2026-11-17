"""Deep supervised neural inversion network under Threat Model 3 (ISO/IEC 30136)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemplateInverterMLP(nn.Module):
    """Deep Multi-Layer Perceptron trained to invert protected templates t -> e."""

    def __init__(self, dim_in: int, dim_out: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        e_hat = self.net(t)
        return F.normalize(e_hat, p=2, dim=-1)


def train_neural_inverter(
    templates_train: np.ndarray,
    embeddings_train: np.ndarray,
    templates_val: np.ndarray,
    embeddings_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[TemplateInverterMLP, float]:
    """Trains a supervised neural inverter to reconstruct continuous embeddings e from t.

    Loss function:
        L = MSE(e_hat, e) + 0.5 * (1 - cosine_similarity(e_hat, e))

    Returns
    -------
    tuple[TemplateInverterMLP, float]
        Trained model and final validation mean cosine similarity.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    dim_in = templates_train.shape[1]
    dim_out = embeddings_train.shape[1]

    model = TemplateInverterMLP(dim_in, dim_out).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t_train_t = torch.from_numpy(templates_train).to(device)
    e_train_t = torch.from_numpy(embeddings_train).to(device)
    t_val_t = torch.from_numpy(templates_val).to(device)
    e_val_t = torch.from_numpy(embeddings_val).to(device)

    dataset = torch.utils.data.TensorDataset(t_train_t, e_train_t)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )

    for epoch in range(epochs):
        model.train()
        for t_b, e_b in loader:
            optimizer.zero_grad()
            e_hat = model(t_b)
            loss_mse = F.mse_loss(e_hat, e_b)
            loss_cos = 1.0 - F.cosine_similarity(e_hat, e_b, dim=-1).mean()
            loss = loss_mse + 0.5 * loss_cos
            loss.backward()
            optimizer.step()

    # Validation evaluation
    model.eval()
    with torch.no_grad():
        e_hat_val = model(t_val_t)
        val_cos_sim = float(F.cosine_similarity(e_hat_val, e_val_t, dim=-1).mean().item())

    return model, val_cos_sim

