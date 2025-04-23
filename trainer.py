import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from logger import logger
import numpy as np

class Trainer:
    def __init__(self, model, dataset, cfg):
        self.model = model.to(cfg.DEVICE)
        self.dataset = dataset
        self.cfg = cfg
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.LR)
        self.criterion = nn.MSELoss()

    def create_disjoint_masks(self, data):
        batch_size, num_features = data.shape
        indices = torch.randperm(num_features)
        split_idx = int(num_features * self.cfg.MASK_MIN_CONTEXT)
        context_indices = indices[:split_idx]
        target_indices = indices[split_idx:]
        context_mask = torch.zeros_like(data, dtype=torch.bool)
        target_mask = torch.zeros_like(data, dtype=torch.bool)
        context_mask[:, context_indices] = True
        target_mask[:, target_indices] = True
        return context_mask, target_mask

    def train_epoch(self):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True)
        epoch_loss = 0
        self.model.train()

        for batch in tqdm(loader):
            batch = batch.to(self.cfg.DEVICE)
            context_mask, target_mask = self.create_disjoint_masks(batch)
            context_x = batch.clone()
            context_x[~context_mask] = 0
            target_x = batch.clone()
            target_x[~target_mask] = 0
            with torch.no_grad():
                target_repr = self.model(target_x, is_target=True)
            pred_repr = self.model(context_x, is_target=False)
            self.optimizer.zero_grad()
            loss = self.criterion(pred_repr, target_repr)
            loss.backward()
            self.optimizer.step()
            self.model.update_target_encoder()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch loss: {avg_loss:.4f}")
        return avg_loss 