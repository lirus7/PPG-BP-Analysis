import argparse
import os
import random
import sys
import time

import numpy as np

import ite
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


"""
Simple 5 layer perceptron autoencoder. 
"""


class Net(nn.Module):
    def __init__(self, input_size=45, hidden_size=45, bottleneck_size=20):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size

        # ENCODER
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.bottleneck_size)

        # DECODER
        self.fc4 = torch.nn.Linear(self.bottleneck_size, self.hidden_size)
        self.fc5 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6 = torch.nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x):
        enc = F.leaky_relu(self.fc1(x), 0.1)
        enc = F.leaky_relu(self.fc2(enc), 0.1)
        enc = F.leaky_relu(self.fc3(enc), 0.1)

        dec = F.leaky_relu(self.fc4(enc), 0.1)
        dec = F.leaky_relu(self.fc5(dec), 0.1)
        dec = F.leaky_relu(self.fc6(dec), 0.1)
        return enc, dec


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ppg_path",
    help="Path of Normalized PPG numpy array with each element representing the PPG signal",
    default="../data/flatten_ppg.npy",
)

parser.add_argument(
    "-label_path",
    help="Path of PPG numpy array with each element representing the PPG signal",
    default="../data/flatten_sbp.npy",
)

parser.add_argument("-ite_path", help="Path to the ite-repo", default=None)
parser.add_argument(
    "-emb_dim", help="size of the bottleneck layer embeddings", default=20
)

parser.add_argument(
    "-neighbours",
    help="neighbour parameter for the estimation of mutual information. Higher values reduce the variance but might introduce a bias",
    default=100,
)

parser.add_argument("-device", help="GPU or CPU", default="cuda:0")


args = parser.parse_args()

# Loading the ite-library for mutual information estimation. Check https://bitbucket.org/szzoli/ite-in-python/
sys.path.insert(
    1,
    "/home/t-surilmehta/home/szzoli-ite-in-python-44a8f15e2dc9/szzoli-ite-in-python-44a8f15e2dc9",
)

# Loading data and parameters
ppg = np.load(f"{args.ppg_path}", allow_pickle=True)
label = np.load(f"{args.label_path}", allow_pickle=True)
emb_dim = args.emb_dim
device = args.device
input_size = ppg.shape[1]

# Initializing model and optimizers
model = Net(input_size, input_size, emb_dim).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-5)

# Conversion to tensors
X, Y = torch.FloatTensor(ppg), torch.FloatTensor(label)
train_dataset = TensorDataset(X, Y)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

epochs = 10000

# Training+Validation loop
for epoch in range(epochs + 1):
    optim.zero_grad()
    train_loss, test_loss = 0, 0
    model.train()
    for inp, label in train_loader:
        inp, label = inp.to(device), label.to(device)
        with torch.set_grad_enabled(True):
            enc, out = model(inp)
            loss = F.mse_loss(out, inp)
            loss.backward()
            optim.step()
        train_loss += loss.item()

    model.eval()
    y, encodings = [], []
    for inp, label in train_loader:
        inp, label = inp.to(device), label.to(device)
        with torch.set_grad_enabled(False):
            enc, out = model(inp)
            loss = F.mse_loss(out, inp)
            encodings.extend(enc.cpu().detach().numpy())
            y.extend(label.cpu().detach().numpy())
        test_loss += loss.item()

    train_loss, test_loss = (
        round(train_loss / len(train_loader), 3),
        round(test_loss / len(train_loader), 3),
    )

    # Logging the loss values and estimating mutual information.
    if epoch % 100 == 0:
        print(f"Epoch:{epoch} Train_Loss:{train_loss} Test_Loss:{test_loss}")

    if train_loss < 0.1 or test_loss < 0.1 and epoch % 5 == 0:
        print(f"Epoch:{epoch} Train_Loss:{train_loss} Test_Loss:{test_loss}")
        encodings, y = np.asarray(encodings), np.asarray(y).reshape(-1, 1)
        co = ite.cost.MIShannon_DKL(
            mult=True, kl_co_name="BDKL_KnnK", kl_co_pars={"k": args.neighbours}
        )
        val = co.estimation(np.hstack((encodings, y)), [emb_dim, 1])
        print(f"MI value:{val}")
