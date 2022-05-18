import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

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

window = 48
batch_size=32
num_epochs = 5
ADD_WEIGHTS = True
PATH = 'model_wo_aqi_w_weights_48.pt'

###################################
#     Functions preprocessing     #
###################################

data = pd.read_csv('../processed_data/full_data_imputed_with_EAQI.csv')

data = data.loc[data["Jahr"] < 2021]

X_labels = ['Zweirad', 'Personenwagen', 'Lastwagen', 'Hr', 'RainDur', 'T', 'WVs', 'StrGlo', 'p'] 
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


def train_model(model, loss_function, optimizer, train_data_full, num_epochs=num_epochs):

    torch.backends.cudnn.benchmark = True
    best_acc = 0.0
    stats = {}

    for epoch in range(num_epochs):

        no_train_triplets = int(len(train_data_full)*0.7)
        no_val_triplets = len(train_data_full) - no_train_triplets
        print(f"train no: {no_train_triplets}")
        print(f"valid no: {no_val_triplets}")
        train_data, valid_data = random_split(train_data_full, [no_train_triplets, no_val_triplets], generator=torch.Generator().manual_seed(30+epoch))
        train_loader = DataLoader(train_data, batch_size=32,
                           shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=32,
                           shuffle=True)


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
    def __init__(self, hidden_size=48, num_layers=1):
        super(LSTM_Model, self).__init__()

        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=len(X_labels), hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(48, 16)
        self.fc2 = nn.Linear(16, 4) 

        self.relu = nn.ReLU()
    
    def forward(self,x):
        # Propagate input through LSTM
        x = x.type(torch.DoubleTensor)

        output, (hn, cn) = self.lstm(x) 
        hn = hn.view(-1, self.hidden_size)
        x = self.relu(hn) 
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x) 
        x = self.dropout(x)
        x = self.fc2(x) 
        return x

######################################
#         Training the model         #
######################################

if __name__ == "__main__":

    y_data.loc[y_data == 4.0] = 3.0

    X, y = create_windows(X_data, y_data, window)

    if(ADD_WEIGHTS):
        count = Counter(y)
        count_list = [int(cnt) for cnt in count.values()]
    
        min = np.max(np.array(count_list))
        ins_weights = torch.tensor([min/cnt for cnt in count_list])[:4]
        print(ins_weights)

    whole_data = SeqDataset(X, y)

    train_loader = DataLoader(whole_data, batch_size=batch_size,
                            shuffle=True)
  
    model = LSTM_Model()

    loss_f = torch.nn.CrossEntropyLoss()
    if(ADD_WEIGHTS):
        loss_f = torch.nn.CrossEntropyLoss(weight=ins_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model = model.double()


    best_model, stats = train_model(model, loss_f, optimizer, whole_data)

    