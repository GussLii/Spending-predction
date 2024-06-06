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
from tqdm import tqdm

from data_Loader import loader
from model import TransactionPredictionModel
import matplotlib.pyplot as plt

train_loader, test_loader = loader(32, True, True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_test(model, train_loader, test_loader, num_epochs, learning_rate):
    # Define optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    model = model.to(device)

    train_losses = []
    test_losses = []
    test_accuracies = []

    # Training loop
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for data in train_loader:
            brand_indices, time_features, amounts, date_labels, merchant_labels = data
            # Move data to the device
            brand_indices = brand_indices.to(device)
            time_features = time_features.to(device)
            amounts = amounts.to(device)
            date_labels = date_labels.to(device)
            merchant_labels = merchant_labels.to(device)

            brand_indices = brand_indices.long()

            optimizer.zero_grad()

            amount_output, date_output, merchant_output = model(brand_indices, time_features, amounts.float())

            loss_amount = criterion_mse(amount_output.squeeze(), amounts)
            loss_date = criterion_mse(date_output.squeeze(), date_labels)
            loss_merchant = criterion_ce(merchant_output, merchant_labels)

            total_loss = loss_amount + loss_date + 0.5 * loss_merchant
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Average Training Loss: {avg_train_loss}')

        # Evaluation loop
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for data in test_loader:
                brand_indices, time_features, amounts, date_labels, merchant_labels = data
                # Move data to the device
                brand_indices = brand_indices.to(device)
                time_features = time_features.to(device)
                amounts = amounts.to(device)
                date_labels = date_labels.to(device)
                merchant_labels = merchant_labels.to(device)

                brand_indices = brand_indices.long()

                amount_output, date_output, merchant_output = model(brand_indices, time_features, amounts.float())

                loss_amount = criterion_mse(amount_output.squeeze(), amounts)
                loss_date = criterion_mse(date_output.squeeze(), date_labels)
                loss_merchant = criterion_ce(merchant_output, merchant_labels)

                total_loss += (loss_amount + loss_date + 0.5 * loss_merchant).item()

                _, predicted = torch.max(merchant_output, 1)
                correct_predictions += (predicted == merchant_labels).sum().item()
                total_predictions += merchant_labels.size(0)

        avg_test_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions
        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Average Test Loss: {avg_test_loss}, Test Accuracy: {accuracy}')

    # Plotting the loss and accuracy
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epochs')
    
    plt.tight_layout()
    plt.show()

num_brands = 1500
brand_embedding_dim = 8
time_feature_dim = 32
nhead = 2
num_decoder_layers = 3
d_model = 32
model=TransactionPredictionModel(num_brands, brand_embedding_dim, time_feature_dim, nhead, num_decoder_layers, d_model)
train_test(model, train_loader, test_loader, num_epochs=5, learning_rate=0.001)