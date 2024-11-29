# This program is created by Chanwoo Yoon and Juwon Lee

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
        self.linear_relu_stack = None
        self.layers = ""

    def set_layers(self, layers, activation):
        self.layers = layers
        # set activation function for part2 extracredit
        if activation == "ReLU":
            activation_fn = nn.ReLU()
        elif activation == "Tanh":
            activation_fn = nn.Tanh()
        else:
            activation_fn = nn.SiLU()
        # set layers
        if layers == "small":
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 10),
                activation_fn,
                nn.Linear(10, 10)
            )
        if layers == "medium":
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 10)
            )
        if layers == "big":
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 10)
            )



    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def get_layers(self):
        return self.layers

class NNProcessor():
    # this class processes our NN(Neural Network). this class is capable of training and validating the given data
    def __init__(self, id):
        self.model = NeuralNetwork().to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        self.train_set = None
        self.val_set = None
        self.train_loss = 0
        self.val_loss = 0
        self.train_num_batches = 0
        self.val_num_batches = 0
        self.id = id
        self.train_accuracy = 0
        self.val_accuracy = 0

    def train(self):
        self.model.train()
        self.train_num_batches = len(self.train_set)
        self.train_loss = 0
        total_loss = 0
        train_correct = 0
        size = len(self.train_set.dataset)
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
            self.train_loss += loss.item()
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        self.train_accuracy = train_correct/size*100

    def validate(self):
        self.model.eval()
        self.val_num_batches = len(self.val_set)
        self.val_loss = 0
        val_correct = 0
        size = len(self.val_set.dataset)
        with torch.no_grad():
            for batch, (X, y) in enumerate(self.val_set):
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.val_loss += loss.item()
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            self.val_accuracy = val_correct/size*100

    def set_data(self, train_dataloader, val_dataloader):
        self.train_set = train_dataloader
        self.val_set = val_dataloader

    def get_id(self):
        return self.id

    def get_train_loss(self):
        return self.train_loss/self.train_num_batches

    def get_val_loss(self):
        return self.val_loss/self.val_num_batches

    def get_train_accuracy(self):
        return self.train_accuracy

    def get_val_accuracy(self):
        return self.val_accuracy

    def set_layers(self, layers, activation):
        self.model.set_layers(layers, activation)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-1) # lr: learning rate

    def get_parameters(self):
        return self.model.state_dict()

    def get_layers(self):
        return self.model.get_layers()
# part2
def run_epoch_different_architecture(training_data, epochs, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    NNProcessors = [[], [], []]
    # for each list in NNProcessors, put k object of NNP(NNProcessor class)
    for i in range(len(NNProcessors)):
        for j in range(k):
            NNP = NNProcessor(j)
            NNProcessors[i].append(NNP)

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(training_data)))):
        train_subset = Subset(training_data, train_idx)
        val_subset = Subset(training_data, val_idx)
        # split our data into actual training and validation sets
        train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=64, shuffle=False)
        # feed two splitted data to corresponding ProcessFold() object
        NNProcessors[0][fold].set_data(train_dataloader, val_dataloader)
        NNProcessors[1][fold].set_data(train_dataloader, val_dataloader)
        NNProcessors[2][fold].set_data(train_dataloader, val_dataloader)
    # a list to store NNPs with three different structure
    models = [[], [], []]
    layers = ["medium", "medium", "medium"]
    activation = ["ReLU", "Tanh", "Swish"]

    for i in range(len(models)):
        for j in range(3):
            NNP = NNProcessors[i][j]
            models[i].append(NNP)

    for i in range(len(models)):
        for j in range(len(models[i])):
            models[i][j].set_layers(layers[i], activation[i])


    train_accuracies = [[], [], []]
    val_accuracies = [[], [], []]
    smallest_validation_losses = [None, None, None]
    best_model_parameters = [None, None, None]
    result = []
    for e in range(epochs):
        # this list stores three pairs of training loss and validation loss
        # the three pairs correspond to the thee models with different architectures
        sub_result = []
        for j in range(len(models)): # 3
            training_losses = []
            validation_losses = []
            for i in range(len(models[j])): # 3
                models[j][i].train()
                models[j][i].validate()
                if smallest_validation_losses[j] == None or smallest_validation_losses[j] > models[j][i].get_val_loss():
                    smallest_validation_losses[j] = models[j][i].get_val_loss()
                    best_model_parameters[j] = models[j][i].get_parameters()
                training_losses.append(models[j][i].get_train_loss())
                validation_losses.append(models[j][i].get_val_loss())
                train_accuracies[j].append(models[j][i].get_train_accuracy())
                val_accuracies[j].append(models[j][i].get_val_accuracy())
            sub_result.append((training_losses, validation_losses))
        result.append(sub_result)
        print("Epoch " + str(e+1) + " done")
    print("All " + str(epochs) + " epochs are done")
    best_val_accuracies = [[], [], []]
    for i in range(len(val_accuracies)):
        sorted_list = sorted(val_accuracies[i])[::-1]
        best_val_accuracies[i].append(sorted_list[0])
        best_val_accuracies[i].append(sorted_list[1])
        best_val_accuracies[i].append(sorted_list[2])
    for i in range(3):
        print("\n" + layers[i] + " model with " + activation[i] + " non-linear function: ")
        print("Average training accuracy-> " + str(round(sum(train_accuracies[i])/len(train_accuracies[i]), 2)) + "%")
        print("Standard error of the mean of the training accuracy-> " + str(round(np.std(train_accuracies[i]), 2)) + "\n")
        print("Average validation accuracy-> " + str(round(sum(val_accuracies[i])/len(val_accuracies[i]), 2)) + "%")
        print("Standard error of the mean of the validation accuracy-> " + str(round(np.std(val_accuracies[i]), 2)) + "\n")
        print("Average validation accuracy for the best model for each validation set-> " + str(round(sum(best_val_accuracies[i])/len(best_val_accuracies[i]), 2)) + "%")
        print("Standard error of the mean of the validation accuracy of the best models-> " + str(round(np.std(best_val_accuracies[i]), 2)) + "\n")
    return result, best_model_parameters


def plot_different_architecture(result):
    epochs_range = range(len(result))
    average_train_losses = [[], [], []]
    average_val_losses = [[], [], []]
    std_train = [[], [], []]
    std_val = [[], [], []]
    for epoch in result:
        for i in range(len(epoch)):
            average_train_losses[i].append(sum(epoch[i][0])/len(epoch[i][0]))
            average_val_losses[i].append(sum(epoch[i][1])/len(epoch[i][1]))
            std_train[i].append(np.std(epoch[i][0]))
            std_val[i].append(np.std(epoch[i][1]))
    colors_train = ['r', 'g', 'b']
    colors_val = ['pink', 'lightgreen', 'skyblue']

    for i in range(len(average_train_losses)):
        for j in range(len(average_train_losses[i])):
            plt.plot(j, average_train_losses[i][j], 'o', color = colors_train[i])
            plt.plot(j, average_val_losses[i][j], 'o', color = colors_val[i])
    for i in range(len(std_train)):
        plt.fill_between(
            epochs_range,
            [average_train_losses[i][j] - std_train[i][j] for j in range(len(std_train[i]))],
            [average_train_losses[i][j] + std_train[i][j] for j in range(len(std_train[i]))],
            color=colors_train[i],
            alpha=0.2
        )
        plt.fill_between(
            epochs_range,
            [average_val_losses[i][j] - std_val[i][j] for j in range(len(std_val[i]))],
            [average_val_losses[i][j] + std_val[i][j] for j in range(len(std_val[i]))],
            color=colors_val[i],
            alpha=0.2
        )
    plt.title('Red: ReLU, Green: Tanh, Blue: Swish')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()



def run_Epoch(training_data, epochs, k, layers):
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
    result = []
    smaller_NNPs = []
    # we are processing one NNPs instead of k
    smaller_NNPs.append(NNProcessors[0])
    #smaller_NNPs.append(NNProcessors[1])
    #smaller_NNPs.append(NNProcessors[2])
    for NNP in smaller_NNPs:
        NNP.set_layers(layers)

    train_accuracies = []
    val_accuracies0 = []
    val_accuracies1 = []
    val_accuracies2 = []
    #val_accuracies = [val_accuracies0, val_accuracies1, val_accuracies2]
    val_accuracies = [val_accuracies0]
    smallest_validation_loss = None
    best_model_parameter = None
    for e in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        trains = []
        vals = []
        for i in range(len(smaller_NNPs)):
            smaller_NNPs[i].train()
            smaller_NNPs[i].validate()
            if smallest_validation_loss == None or smallest_validation_loss > smaller_NNPs[i].get_val_loss():
                smallest_validation_loss = smaller_NNPs[i].get_val_loss()
                best_model_parameter = smaller_NNPs[i].get_parameters()
            trains.append(smaller_NNPs[i].get_train_loss())
            vals.append(smaller_NNPs[i].get_val_loss())
            total_train_loss += smaller_NNPs[i].get_train_loss()
            total_val_loss += smaller_NNPs[i].get_val_loss()
            train_accuracies.append(smaller_NNPs[i].get_train_accuracy())
            val_accuracies[i].append(smaller_NNPs[i].get_val_accuracy())
        result.append((trains, vals))
        print("Epoch " + str(e+1))
        print("average training loss: " + str(total_train_loss/len(smaller_NNPs)))
        print("average validation loss: " + str(total_val_loss/len(smaller_NNPs)) + "\n")
    vals = []
    for v in val_accuracies:
        for accuracy in v:
            vals.append(accuracy)
    best_val_accuracies = []
    for v in val_accuracies:
        best_val_accuracies.append(max(v))
    print("All " + str(epochs) + " epochs are done")
    print("Average training accuracy-> " + str(round(sum(train_accuracies)/len(train_accuracies), 2)) + "%")
    print("Standard error of the mean of the training accuracy-> " + str(round(np.std(train_accuracies), 2)) + "\n")
    print("Average validation accuracy-> " + str(round(sum(vals)/len(vals), 2)) + "%")
    print("Standard error of the mean of the validation accuracy-> " + str(round(np.std(vals), 2)) + "\n")
    print("Average validation accuracy for the best model for each validation set-> " + str(round(sum(best_val_accuracies)/len(best_val_accuracies), 2)) + "%")
    print("Standard error of the mean of the validation accuracy of the best models-> " + str(round(np.std(best_val_accuracies), 2)) + "\n")
    return result, best_model_parameter


def plot(alist):
    epochs_range = range(len(alist))
    average_train_loss = []
    average_val_loss = []
    std_train = []
    std_val = []
    for i in range(len(alist)):
        average_train_loss.append(sum(alist[i][0])/len(alist[i][0]))
        average_val_loss.append(sum(alist[i][1])/len(alist[i][1]))
        std_train.append(np.std(alist[i][0]))
        std_val.append(np.std(alist[i][1]))
    for i in range(len(alist)):
        plt.plot(i, average_train_loss[i], 'o', color = 'r')
        plt.plot(i, average_val_loss[i], 'o', color = 'b')
    plt.fill_between(
        epochs_range,
        [average_train_loss[i] - std_train[i] for i in range(len(std_train))],
        [average_train_loss[i] + std_train[i] for i in range(len(std_train))],
        color="red",
        alpha=0.2
    )
    plt.fill_between(
        epochs_range,
        [average_val_loss[i] - std_val[i] for i in range(len(std_val))],
        [average_val_loss[i] + std_val[i] for i in range(len(std_val))],
        color="blue",
        alpha=0.2
    )

    plt.title('Red: train loss, Blue: validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def final_check(best_model_parameter, layers):
    model = NeuralNetwork().to(device)
    model.set_layers(layers)
    model.load_state_dict(best_model_parameter)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    training_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    train_loss, train_correct = 0, 0
    test_loss, test_correct = 0, 0
    size = len(training_dataloader.dataset)
    with torch.no_grad():
        for X, y in training_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            train_loss += loss_fn(pred, y).item()
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_correct /= size
    print(f"Training accuracy on the best model-> {(100*train_correct):>0.1f}%")

    size = len(test_dataloader.dataset)
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_correct /= size
    print(f"Testing accuracy on the best model-> {(100*test_correct):>0.1f}%")

def final_check_different_architecture(best_model_parameters):
    layers = ["small", "medium", "big"]
    models = []
    model1 = NeuralNetwork().to(device)
    model2 = NeuralNetwork().to(device)
    model3 = NeuralNetwork().to(device)
    model1.set_layers("medium", "ReLU")
    model2.set_layers("medium", "Tanh")
    model3.set_layers("medium", "Swish")
    model1.load_state_dict(best_model_parameters[0])
    model2.load_state_dict(best_model_parameters[1])
    model3.load_state_dict(best_model_parameters[2])
    model1.eval()
    model2.eval()
    model3.eval()
    models.append(model1)
    models.append(model2)
    models.append(model3)

    loss_fn = nn.CrossEntropyLoss()
    training_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    idx = 0
    for model in models:
        print("\n" + layers[idx] + ": ")
        idx += 1
        train_loss, train_correct = 0, 0
        test_loss, test_correct = 0, 0
        size = len(training_dataloader.dataset)
        with torch.no_grad():
            for X, y in training_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                train_loss += loss_fn(pred, y).item()
                train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_correct /= size
        print(f"Training accuracy on the best model-> {(100*train_correct):>0.1f}%")

        size = len(test_dataloader.dataset)
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_correct /= size
        print(f"Testing accuracy on the best model-> {(100*test_correct):>0.1f}%")

def main():
    layers = "big"
    epochs = 30
    k = 10
    result, best_model_parameter = run_Epoch(training_data, epochs, k, layers)
    final_check(best_model_parameter, layers)
    plot(result)

def main1():
    epochs = 30
    k = 10
    result, best_model_parameters = run_epoch_different_architecture(training_data, epochs, k)
    final_check_different_architecture(best_model_parameters)
    plot_different_architecture(result)

main1()




