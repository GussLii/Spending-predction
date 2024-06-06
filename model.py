
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransactionPredictionModel(nn.Module):
    def __init__(self, num_brands, brand_embedding_dim, time_feature_dim, nhead, num_decoder_layers, d_model):
        super(TransactionPredictionModel, self).__init__()
        self.d_model = d_model
        self.brand_embedding = nn.Embedding(num_brands, brand_embedding_dim)
        self.time_processing = nn.Linear(time_feature_dim, 16) 
        self.amount_processing = nn.Linear(1, 8) 
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=0.1),
            num_layers=num_decoder_layers
        )
        self.amount_predictor = nn.Linear(d_model, 1) 
        self.date_predictor = nn.Linear(d_model, 1) 
        self.merchant_predictor = nn.Linear(d_model, num_brands)

    def forward(self, brand_indices, time_features, amount):
        #print('-----------------------------------')
        #print("Brand indices shape:", brand_indices.shape)

        brand_embeddings = self.brand_embedding(brand_indices).squeeze(1) * math.sqrt(self.d_model)
        brand_embeddings = torch.squeeze(brand_embeddings, 1) if brand_embeddings.dim() > 2 else brand_embeddings
        #print("Adjusted Brand embeddings shape:", brand_embeddings.shape)

        time_features = time_features.unsqueeze(0).expand(32, -1)
        time_embeddings = F.relu(self.time_processing(time_features))
        #print("Time embeddings shape:", time_embeddings.shape)
    
        amount_embeddings = F.relu(self.amount_processing(amount.unsqueeze(1)))
        #print("Amount embeddings shape:", amount_embeddings.shape)

        combined_embeddings = torch.cat([brand_embeddings, time_embeddings, amount_embeddings], dim=-1)
        #print("Combined embeddings shape after adjustment:", combined_embeddings.shape)

        transformer_output = self.transformer_decoder(tgt=combined_embeddings, memory=combined_embeddings)
        #print("Transformer output shape:", transformer_output.shape)
        #print('-----------------------------------')
        amount_output = self.amount_predictor(transformer_output)
        date_output = self.date_predictor(transformer_output)
        merchant_output = F.softmax(self.merchant_predictor(transformer_output), dim=-1)
    
        return amount_output, date_output, merchant_output