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
def load_and_split_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)


    # Convert 'optimized_date' to datetime
    data['optimized_date'] = pd.to_datetime(data['optimized_date'])

    threshold_date = data['optimized_date'].median()

    # Define training and testing periods
    training_end_date = threshold_date
    testing_start_date = training_end_date + pd.Timedelta(days=7)  # Start testing 1 week after the training end date

    # Split data into training and testing datasets
    training_data = data[data['optimized_date'] <= training_end_date]
    testing_data = data[data['optimized_date'] >= testing_start_date]

    return training_data, testing_data


def train_test_split(data, training_end_date, testing_start_date):
    training_data = data[data['optimized_date'] <= training_end_date]
    testing_data = data[data['optimized_date'] >= testing_start_date]
    return training_data, testing_data

def transform_to_sequences(data):
    grouped = data.groupby('member_id').apply(
        lambda x: x.sort_values('optimized_date')
    ).reset_index(drop=True)
    
    sequences = grouped.groupby('member_id').apply(
        lambda x: list(zip(x['optimized_date'], x['merchant_format_name'], x['transaction_amount']))
    )
    
    return sequences

def transform_to_sequences(data):
    grouped = data.groupby('member_id').apply(
        lambda x: x.sort_values('optimized_date')
    ).reset_index(drop=True)
    
    sequences = grouped.groupby('member_id').apply(
        lambda x: list(zip(x['optimized_date'], x['merchant_format_name'], x['transaction_amount']))
    )
    
    return sequences

def prepare_data(sequences):
    merchant_encoder = LabelEncoder()
    all_merchants = [merchant for seq in sequences for _, merchant, _ in seq]
    merchant_encoder.fit(all_merchants)

    day_numbers = []
    merchant_indices = [] 
    labels = []

    for sequence in sequences:
        min_date = min(date for date, _, _ in sequence)
        for date, merchant, amount in sequence:
            day_number = (date - min_date).days
            merchant_index = merchant_encoder.transform([merchant])[0]
            day_numbers.append(day_number)
            merchant_indices.append(merchant_index)
            labels.append(amount)

    return torch.tensor(day_numbers, dtype=torch.float32).to(device), torch.tensor(merchant_indices, dtype=torch.long).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def generate_time_labels(sequences):
    time_labels = []
    for member_id, sequence in sequences.items():
        first_date = sequence[0][0]  
        for transaction in sequence:
            current_date = transaction[0]
            day_diff = (current_date - first_date).days
            time_labels.append(day_diff)
    return torch.tensor(time_labels, dtype=torch.float32).to(device)

def generate_amount_labels(sequences):
    amount_labels = []
    for _, sequence in sequences.items():
        for transaction in sequence:
            amount = transaction[2]
            amount_labels.append(amount)
    
    return torch.tensor(amount_labels, dtype=torch.float32).to(device)

def generate_merchant_labels(sequences):
    merchant_labels = [] 
    for _, sequence in sequences.items():
        for transaction in sequence:
            merchant = transaction[1] 
            merchant_labels.append(merchant)

    from sklearn.preprocessing import LabelEncoder
    merchant_encoder = LabelEncoder()
    merchant_labels_encoded = merchant_encoder.fit_transform(merchant_labels)

    return torch.tensor(merchant_labels_encoded, dtype=torch.long).to(device)

