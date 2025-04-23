import torch
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import entropy
import pacmap

def uniformity(embeddings):
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    sq_pdist = torch.pdist(embeddings, p=2).pow(2)
    return sq_pdist.mul(-2).exp().mean().log().item()

def kl_divergence(embeddings):
    p = embeddings.softmax(dim=-1).mean(dim=0)
    q = torch.full_like(p, 1/p.size(0))
    return entropy(p.cpu(), q.cpu())

def evaluate(model, dataset, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.shape[0]
            reg_tokens = model.reg_token.expand(batch_size, 1, -1)
            x_embed = model.embedding(batch)
            x_with_reg = torch.cat([x_embed, reg_tokens], dim=1)
            emb = model.context_encoder(x_with_reg)
            embeddings.append(emb[:, -1].cpu())  # Use [REG] token representation
    embeddings = torch.cat(embeddings)
    return {
        "uniformity": uniformity(embeddings),
        "kl_divergence": kl_divergence(embeddings),
        "pacmap": pacmap.PaCMAP().fit_transform(embeddings.numpy())
    }
 