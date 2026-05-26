import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.train_mlp import MLP


class DistilledMLPClassifier:
    """MLP student trained with BCE + alpha * MSE(teacher_soft, student_soft)."""

    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3,
                 epochs=100, batch_size=32, lr=1e-3, alpha=0.5,
                 weight_pos=1.0, seed=42):
        torch.manual_seed(seed)
        self.model = MLP(input_dim, hidden_dims, dropout)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.weight_pos = weight_pos
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, teacher_probs_train, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X_train)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        p_teacher = torch.tensor(teacher_probs_train, dtype=torch.float32)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        pos_weight = torch.tensor([self.weight_pos])
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mse_loss = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience = 15
        patience_count = 0

        for epoch in range(self.epochs):
            self.model.train()
            idx = torch.randperm(len(X_t))
            for start in range(0, len(X_t), self.batch_size):
                bi = idx[start:start + self.batch_size]
                xb, yb, pt = X_t[bi], y_t[bi], p_teacher[bi]
                logits = self.model(xb)
                p_student = torch.sigmoid(logits)
                loss = bce_loss(logits, yb) + self.alpha * mse_loss(p_student, pt)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    xv = torch.tensor(self.scaler.transform(X_val), dtype=torch.float32)
                    yv = torch.tensor(y_val, dtype=torch.float32)
                    val_loss = bce_loss(self.model(xv), yv).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        xt = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(xt).numpy()
        prob = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - prob, prob])

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Student model saved: {path}")
