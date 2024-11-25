# this code is created by Chanwoo Yoon and Juwon Lee
# only one pair of training and validation sets are considerd in this program

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import numpy as np

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NNProcessor():
    def __init__(self, id):
        self.model = NeuralNetwork().to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3) # lr: learning rate
        self.train_set = None
        self.val_set = None
        self.train_loss = 0
        self.val_loss = 0
        self.train_num_batches = 0
        self.val_num_batches = 0
        self.id = id
    
    def train(self):
        self.model.train()
        self.train_num_batches = len(self.train_set)
        total_loss = 0
        for batch, (X, y) in enumerate(self.train_set):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # compute training loss and number of correct prediction
            self.train_loss += loss_fn(pred, y).item()
            #train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    def validate(self):
        self.model.eval()
        self.val_num_batches = len(self.val_set)
        with torch.no_grad():
            for batch, (X, y) in enumerate(self.val_set):
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.val_loss += loss_fn(pred, y).item()
                #val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    def set_data(self, train_dataloader, val_dataloader):
        self.train_set = train_dataloader
        self.val_set = val_dataloader

    def get_id(self):
        return self.id

    def get_train_loss(self):
        return self.train_loss/self.train_num_batches
    
    def get_val_loss(self):
        return self.val_loss/self.val_num_batches

    def get_parameters(self):
        return self.model.state_dict()

def run_Epoch(training_data, epochs, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    NNProcessors = []
    for i in range(k):
        NNProcessors.append(NNProcessor(i))

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(training_data)))):
        train_subset = Subset(training_data, train_idx)
        val_subset = Subset(training_data, val_idx)
        # split our data into actual training and validation sets
        train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=64, shuffle=False)
        # feed two splitted data to corresponding ProcessFold() object
        NNProcessors[fold].set_data(train_dataloader, val_dataloader)

    for e in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        a=[]
        for NNP in NNProcessors:
            NNP.train()
            NNP.validate()
            a.append(NNP.get_train_loss())
            total_train_loss += NNP.get_train_loss()
            total_val_loss += NNP.get_val_loss()
        print("Epoch " + str(e+1))
        print(a)
        print("average training loss: " + str(total_train_loss/k))
        print("average validation loss: " + str(total_val_loss/k))

def main():
    epochs = 10
    k = 10
    run_Epoch(training_data, epochs, k)

main()




ß
