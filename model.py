import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.encoder(x)

class SQLJEPAModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, momentum=0.996):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.context_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.target_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.predictor = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.momentum = momentum
        self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def update_target_encoder(self):
        for target_param, context_param in zip(self.target_encoder.parameters(), 
                                             self.context_encoder.parameters()):
            target_param.data = target_param.data * self.momentum + \
                              context_param.data * (1. - self.momentum)

    def forward(self, x, is_target=False):
        batch_size = x.shape[0]
        reg_tokens = self.reg_token.expand(batch_size, 1, -1)
        x_embed = self.embedding(x)
        x_with_reg = torch.cat([x_embed, reg_tokens], dim=1)
        
        if is_target:
            hidden = self.target_encoder(x_with_reg)
            return hidden[:, -1]
        else:
            context_repr = self.context_encoder(x_with_reg)
            predictions = self.predictor(context_repr)
            return predictions[:, -1] 