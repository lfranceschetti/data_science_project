import random

from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from imblearn.combine import SMOTEENN 
import torch
import numpy as np
import pandas as pd
import pickle
from collections import Counter

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# Making sure that results are reproducable
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


###################################
#        Model configuaration     #
###################################

TRAIN = False
window = 48
batch_size=32
num_epochs = 12
PATH = 'model_w_weights.pt'

###################################
#     Functions preprocessing     #
###################################

data = pd.read_csv('../processed_data/full_data_imputed_with_EAQI.csv')

data = data.loc[data["Jahr"] < 2021]
data = data.loc[data["AQI"] < 4.0]

X_labels = ['Zweirad', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p', "AQI"] 
y_label = "AQI"

X_data = data[X_labels]
y_data = data[y_label]


def create_windows(x_data, y_data, window):
    X = []
    y = []
    for i in range(x_data.shape[0]):
        if(i >= window):
            indices = range(i - window, i)
            X.append(x_data.iloc[indices])
            y.append(y_data.iloc[i])
    return np.array(X ,dtype=np.double), np.array(y, dtype=np.double)



class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx])
        label = torch.tensor(int(self.labels[idx]))
        label_one_hot = F.one_hot(label, num_classes=4) 
        return seq, label_one_hot


######################################
#       Functions for training       #
######################################


def accuracy(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # computes the classification accuracy
    correct_label = torch.argmax(
        output, axis=-1) == torch.argmax(label, axis=-1)
    assert correct_label.shape == (output.shape[0],)
    acc = torch.mean(correct_label.float())
    assert 0. <= acc <= 1.
    return acc


def train_model(model, loss_function, optimizer, train_loader, valid_loader, num_epochs=num_epochs):

    torch.backends.cudnn.benchmark = True
    best_acc = 0.0
    stats = {}

    for epoch in range(num_epochs):
        print(f"Start epoch: {epoch}")
        stats_epoch = {}

        for phase in ['train', 'val']:
            stats_phase = {"n_epoch": 0, "loss": 0.0, "acc": 0.0}

            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader

            for inputs, labels in dataloader:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                   
                    loss = loss_function(outputs.float(), labels.float())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    n_batch = inputs.shape[0]
                    stats_phase["n_epoch"] += n_batch
                    stats_phase["acc"] += accuracy(outputs, labels) * n_batch
                    stats_phase["loss"] += loss * n_batch

            stats_phase["acc"] = stats_phase["acc"] / \
                stats_phase["n_epoch"]
            stats_phase["loss"] = stats_phase["loss"] / \
                stats_phase["n_epoch"]
            stats_epoch[phase] = stats_phase

            if phase == 'val' and stats_phase["acc"] > best_acc:
                best_acc = stats_phase["acc"]
                torch.save(model.state_dict(), PATH)
    
        print(stats_epoch)
        stats[f"Epoch {epoch}"] = stats_epoch

    model.load_state_dict(torch.load(PATH))
    return model, stats

######################################
#    Custom Network architecture     #
######################################

class LSTM_Model(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super(LSTM_Model, self).__init__()

        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=len(X_labels), hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(64, 4) 

        self.relu = nn.ReLU()
    
    def forward(self,x):
        # Propagate input through LSTM
        x = x.type(torch.DoubleTensor)

        output, (hn, cn) = self.lstm(x) 
        hn = hn.view(-1, self.hidden_size)
        x = self.relu(hn) 
        x = self.dropout(x)
        x = self.fc(x) 
        return x


######################################
#         Training the model         #
######################################

if __name__ == "__main__":

    X, y = create_windows(X_data, y_data, window)


    # print(Counter(y))
    # print(X.shape)
    # X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    # print(X.shape)

    # strategy = {0.0:60810, 1.0:51573, 2.0:10000, 3.0:10000, 4.0:1}
    # oversample = SMOTEENN(random_state = 42, sampling_strategy=strategy)
    # X, y = oversample.fit_resample(X, y)

    # X = np.reshape(X, (X.shape[0], 48, 10))  
    # print(Counter(y))
 
    # with open('test.npy', 'wb') as f:
    #     np.save(f, X)
    #     np.save(f, y)
    # with open('test.npy', 'rb') as f:   
    #     X = np.load(f)
    #     y = np.load(f)

    # print(X.shape)
    # print(y.shape)


    # train_data = SeqDataset(X[:int(0.7*X.shape[0])-11], y[:int(0.7*X.shape[0])-11])

    # valid_data = SeqDataset(X[int(0.7*X.shape[0])-11:-25], y[int(0.7*X.shape[0])-11:-25])

    # train_loader = DataLoader(train_data, batch_size=batch_size,
    #                         shuffle=True)
    # valid_loader = DataLoader(valid_data, batch_size=batch_size,
    #                         shuffle=True)

    y = [val-1.0 for val in y if val == 4.0]
    X, y = create_windows(X_data, y_data, window)

    count = Counter(y)
    print(count)
    ins_weights = torch.tensor([1/int(cnt) for cnt in count.values()])[:4]
    print(ins_weights)
    ins_weights

    train_data = SeqDataset(X[:int(0.7*X.shape[0])-22], y[:int(0.7*X.shape[0])-22])

    valid_data = SeqDataset(X[int(0.7*X.shape[0])-22:-18], y[int(0.7*X.shape[0])-22:-18])


    train_loader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size,
                            shuffle=True)


    model = LSTM_Model()

    loss_f = torch.nn.CrossEntropyLoss(weight=ins_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    model = model.double()


    best_model, stats = train_model(model, loss_f, optimizer, train_loader, valid_loader)

    with open('saved_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)



