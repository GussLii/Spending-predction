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

from data_preprocessing import load_and_split_data, transform_to_sequences, prepare_data, generate_amount_labels, generate_time_labels, generate_merchant_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loader(bz,sf,dl):
    training_data, testing_data = load_and_split_data('data.csv')
    train_sequences=transform_to_sequences(training_data)
    features_train, merchant_indices_train, labels_train = prepare_data(train_sequences)
    amount_labels_train = generate_amount_labels(train_sequences)
    time_labels_train = generate_time_labels(train_sequences)
    merchant_label_train= generate_merchant_labels(train_sequences)

    train_dateset=TensorDataset(features_train, labels_train, amount_labels_train, time_labels_train, merchant_label_train)
    train_loader=DataLoader(train_dateset, batch_size=bz, shuffle=sf, drop_last=dl)

    testing_sequences=transform_to_sequences(testing_data)
    features_test, merchant_indices_test, labels_test = prepare_data(testing_sequences)
    amount_labels_test = generate_amount_labels(testing_sequences)
    time_labels_test = generate_time_labels(testing_sequences)
    merchant_label_test= generate_merchant_labels(testing_sequences)
    test_dataset=TensorDataset(features_test, labels_test, amount_labels_test, time_labels_test, merchant_label_test)
    test_loader=DataLoader(test_dataset, batch_size=bz, shuffle=sf, drop_last=dl)
    return train_loader, test_loader

#t,s=loader(32, True, True)
#print(len(t), len(s))

