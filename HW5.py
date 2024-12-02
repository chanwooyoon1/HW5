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
from torchvision import transforms
from torchvision import models

'''transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),  # ResNet/VGGに適したサイズ
    transforms.ToTensor()
])'''
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
#-----------------------------------------------------------------------------#
# Part 4
'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Download CIFAR-10 training data
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

# Download CIFAR-10 test data
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)
'''
#-----------------------------------------------------------------------------#

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
        if activation == "ReLU":
            activation_fn = nn.ReLU()
        elif activation == "Tanh":
            activation_fn = nn.Tanh()
        else:
            activation_fn = nn.SiLU()
        if layers == "small":
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(32 * 32 * 3, 10),
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
                nn.Linear(32 * 32 * 3, 512),
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

    def set_model(self, model_name):
        if model_name == "resnet":
            self.model = models.resnet18(pretrained=True)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 10)


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
#-----------------------------------------------------------------------------#
# Part 1
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
    smaller_NNPs.append(NNProcessors[1])
    smaller_NNPs.append(NNProcessors[2])
    for NNP in smaller_NNPs:
        NNP.set_layers(layers, "ReLU")

    train_accuracies = []
    val_accuracies0 = []
    val_accuracies1 = []
    val_accuracies2 = []
    val_accuracies = [val_accuracies0, val_accuracies1, val_accuracies2]
    #val_accuracies = [val_accuracies0]
    smallest_validation_loss = None
    best_model_parameter = None
    stop_counter = 0
    for e in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        trains = []
        vals = []
        prev = None
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
        average_val_loss = sum(vals)/len(vals)
        # early stopping
        '''if prev == None:
            prev = average_val_loss
        if prev != None and average_val_loss > prev:
            stop_counter += 1
        elif prev != None and average_val_loss < prev:
            stop_counter = 0
        if stop_counter == 5:
            print("Early stopping")
            break'''
        prev = average_val_loss
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


def plot(alist, k):
    # this function plots for part 1
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
    '''for i in range(len(alist)):
        plt.plot(i, average_train_loss[i], 'o', color = 'r')
        plt.plot(i, average_val_loss[i], 'o', color = 'b')'''
    train = range(len(average_train_loss))
    val = range(len(average_val_loss))
    plt.plot(train, average_train_loss, color = 'r', label = 'training loss')
    plt.plot(val, average_val_loss, color = 'b', label = 'validation loss')

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

    plt.title('Training and Validation loss vs epoch with ' + str(k) + " folds")
    plt.legend(loc='upper left', fontsize='small')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def final_check(best_model_parameter, layers):
    # this function applies the best parameter to a model and calculates training and testing accuracy
    model = NeuralNetwork().to(device)
    model.set_layers(layers, "ReLU")
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

def main():
    layers = "medium"
    epochs = 60
    k = 20
    result, best_model_parameter = run_Epoch(training_data, epochs, k, layers)
    final_check(best_model_parameter, layers)
    plot(result, k)
main()

## Part 1 end
#-----------------------------------------------------------------------------#




#-----------------------------------------------------------------------------#
# Part 2
def run_epoch_different_architecture(training_data, epochs, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    NNProcessors = [[], [], []]
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
            total_train_accuracy = []
            total_val_accuracy = []
            for i in range(len(models[j])): # 3
                models[j][i].train()
                models[j][i].validate()
                if smallest_validation_losses[j] == None or smallest_validation_losses[j] > models[j][i].get_val_loss():
                    smallest_validation_losses[j] = models[j][i].get_val_loss()
                    best_model_parameters[j] = models[j][i].get_parameters()
                training_losses.append(models[j][i].get_train_loss())
                validation_losses.append(models[j][i].get_val_loss())
                total_train_accuracy.append(models[j][i].get_train_accuracy())
                total_val_accuracy.append(models[j][i].get_val_accuracy())
            train_accuracies[j].append(total_train_accuracy)
            val_accuracies[j].append(total_val_accuracy)
            sub_result.append((training_losses, validation_losses))
        result.append(sub_result)
        print("Epoch " + str(e+1) + " done")
    print("All " + str(epochs) + " epochs are done")
    best_val_accuracies = [[], [], []]
    trains = [[], [], []]
    vals = [[], [], []]
    for i in range(len(train_accuracies)):
        for j in range(len(train_accuracies[i])):
            for k in range(len(train_accuracies[i][j])):
                trains[i].append(train_accuracies[i][j][k])
                vals[i].append(val_accuracies[i][j][k])
    for i in range(len(vals)):
        sorted_list = sorted(vals[i])[::-1]
        best_val_accuracies[i].append(sorted_list[0])
        best_val_accuracies[i].append(sorted_list[1])
        best_val_accuracies[i].append(sorted_list[2])
    for i in range(3):
        print("\n" + layers[i] + " model with " + activation[i] + " non-linear function: ")
        print("Average training accuracy-> " + str(round(sum(trains[i])/len(trains[i]), 2)) + "%")
        print("Standard error of the mean of the training accuracy-> " + str(round(np.std(train_accuracies[i]), 2)) + "\n")
        print("Average validation accuracy-> " + str(round(sum(vals[i])/len(vals[i]), 2)) + "%")
        print("Standard error of the mean of the validation accuracy-> " + str(round(np.std(val_accuracies[i]), 2)) + "\n")
        print("Average validation accuracy for the best model for each validation set-> " + str(round(sum(best_val_accuracies[i])/len(best_val_accuracies[i]), 2)) + "%")
        print("Standard error of the mean of the validation accuracy of the best models-> " + str(round(np.std(best_val_accuracies[i]), 2)) + "\n")
    print(train_accuracies)
    print(val_accuracies)
    return result, best_model_parameters, train_accuracies, val_accuracies

def plot_different_architecture_accuracies(train_accuracies, val_accuracies):
    # this function plots for part 2 accuracies
    epochs_range = range(len(train_accuracies[0]))
    average_train_accuracies = [[], [], []]
    average_val_accuracies = [[], [], []]
    std_train = [[], [], []]
    std_val = [[], [], []]
    for i in range(len(train_accuracies)):
        for j in range(len(train_accuracies[i])):
            average_train_accuracies[i].append(sum(train_accuracies[i][j])/len(train_accuracies[i][j]))
            average_val_accuracies[i].append(sum(val_accuracies[i][j])/len(val_accuracies[i][j]))
            std_train[i].append(np.std(train_accuracies[i][j]))
            std_val[i].append(np.std(val_accuracies[i][j]))
    colors_train = ['r', 'g', 'b']
    layers = ["medium", "medium", "medium"]
    activation = ["ReLU", "Tanh", "Swish"]
    for i in range(len(average_train_accuracies)):
        trains = range(len(average_train_accuracies[i]))
        vals = range(len(average_val_accuracies[i]))
        plt.plot(trains, average_train_accuracies[i], color = colors_train[i], label = 'training accuracy of ' + activation[i] + ' model')
        plt.plot(vals, average_val_accuracies[i], linestyle='--', color = colors_train[i], label = 'validation accuracy of ' + activation[i] + ' model')

    for i in range(len(std_train)):
        plt.fill_between(
            epochs_range,
            [average_train_accuracies[i][j] - std_train[i][j] for j in range(len(std_train[i]))],
            [average_train_accuracies[i][j] + std_train[i][j] for j in range(len(std_train[i]))],
            color=colors_train[i],
            alpha=0.2
        )
        plt.fill_between(
            epochs_range,
            [average_val_accuracies[i][j] - std_val[i][j] for j in range(len(std_val[i]))],
            [average_val_accuracies[i][j] + std_val[i][j] for j in range(len(std_val[i]))],
            color=colors_train[i],
            alpha=0.2
        )
    plt.legend(loc='lower right', fontsize='small')
    plt.title('Accuracy vs Epoch with different non-linear layers')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

def plot_different_architecture(result):
    # this function plots for part 2 loss
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
    layers = ["medium", "medium", "medium"]
    activation = ["ReLU", "Tanh", "Swish"]
    for i in range(len(average_train_losses)):
        trains = range(len(average_train_losses[i]))
        vals = range(len(average_val_losses[i]))
        plt.plot(trains, average_train_losses[i], color = colors_train[i], label = 'training loss of ' + activation[i] + ' model')
        plt.plot(vals, average_val_losses[i], linestyle='--', color = colors_train[i], label = 'validation loss of ' + activation[i] + ' model')

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
            color=colors_train[i],
            alpha=0.2
        )
    plt.legend(loc='upper left', fontsize='small')
    plt.title('Loss vs Epoch with CIFAR-10 data set')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def final_check_different_architecture(best_model_parameters):
    # this function applies the best parameters to a model and calculates training and testing accuracies
    layers = ["medium", "medium", "medium"]
    activation = ["ReLU", "Tanh", "Swish"]
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
        print("\n" + layers[idx] + " with " + activation[idx])
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

def main_part2():
    epochs = 30
    k = 10
    result, best_model_parameters, train_accuracies, val_accuracies = run_epoch_different_architecture(training_data, epochs, k)
    final_check_different_architecture(best_model_parameters)
    plot_different_architecture(result)
#main_part2()
## Part 2 end
#-----------------------------------------------------------------------------#



#-----------------------------------------------------------------------------#
# Part 3
def visualize_first_layer(model):
    first_layer_weights = model.linear_relu_stack[0].weight.data.cpu().numpy()
    return first_layer_weights

def plot_weights(weights, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(weights[i].reshape(28, 28), cmap='viridis')
        ax.axis('off')
    plt.show()

def visualize_1st_layer():
    # this function visualizes the first layer
    layers = "medium"
    epochs = 30
    k = 10
    result, best_model_parameter = run_Epoch(training_data, epochs, k, layers)

    model = NeuralNetwork().to(device)
    model.set_layers("medium", "ReLU")
    model.load_state_dict(best_model_parameter)
    first_layer_weights = visualize_first_layer(model)
    plot_weights(first_layer_weights, num_images=10)


def visualize_first_layer(model):
    first_layer_weights = model.linear_relu_stack[0].weight.data.cpu().numpy()
    return first_layer_weights

def plot_weights(weights, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    for i in range(num_images):
        ax = axes[i]
        weight_image = weights[i].reshape(3, 32, 32).transpose(1, 2, 0)

        weight_image = (weight_image - weight_image.min()) / (weight_image.max() - weight_image.min())

        ax.imshow(weight_image)
        ax.axis('off')
    plt.show()

def visualize_1st_layer():
    layers = "medium"
    epochs = 30
    k = 10
    result, best_model_parameter = run_Epoch(training_data, epochs, k, layers)

    model = NeuralNetwork().to(device)
    model.set_layers("medium", "ReLU")
    model.load_state_dict(best_model_parameter)
    first_layer_weights = visualize_first_layer(model)
    plot_weights(first_layer_weights, num_images=10)
## Part 3 End
#-----------------------------------------------------------------------------#

def accuracyplot():
    train_accuracies = [[[56.49444444444445, 56.71296296296296, 58.12592592592593], [80.8037037037037, 81.10740740740741, 80.7537037037037], [84.67407407407408, 84.58518518518518, 84.51851851851852], [86.1537037037037, 85.95, 85.9962962962963], [87.20185185185186, 87.01296296296296, 87.03333333333333], [87.76851851851852, 87.79074074074073, 87.83703703703704], [88.64814814814815, 88.58148148148149, 88.50555555555556], [88.93148148148148, 89.0851851851852, 88.93333333333334], [89.3425925925926, 89.46481481481482, 89.41111111111111], [89.89259259259259, 89.68703703703703, 89.76851851851852], [90.27407407407408, 90.3425925925926, 90.18333333333334], [90.7462962962963, 90.4462962962963, 90.62407407407407], [90.95370370370371, 90.73703703703704, 90.88518518518518], [91.29814814814814, 91.15925925925926, 91.27037037037037], [91.43518518518519, 91.49074074074073, 91.4074074074074], [91.8425925925926, 91.71851851851852, 91.7537037037037], [92.07777777777778, 91.97777777777777, 91.90555555555555], [92.53518518518518, 92.23333333333333, 92.15185185185184], [92.57407407407408, 92.52037037037037, 92.50185185185185], [92.84259259259258, 92.68333333333332, 92.75925925925927], [93.02777777777777, 92.83148148148148, 92.90555555555555], [93.25925925925927, 93.05, 93.11851851851853], [93.48518518518519, 93.2611111111111, 93.25555555555556], [93.71481481481482, 93.33333333333333, 93.5111111111111], [93.8648148148148, 93.64444444444445, 93.62037037037037], [94.03148148148148, 93.74074074074073, 93.94444444444444], [94.2537037037037, 94.04814814814814, 94.04444444444444], [94.34074074074074, 94.07592592592593, 94.33888888888889], [94.35740740740741, 94.29444444444445, 94.2388888888889], [94.57962962962962, 94.5611111111111, 94.45925925925926]], [[72.30925925925926, 72.3425925925926, 72.67222222222223], [82.16851851851852, 82.05925925925925, 82.03333333333333], [83.90925925925926, 83.9037037037037, 83.66111111111111], [84.79074074074074, 84.85370370370372, 84.60555555555555], [85.39999999999999, 85.41296296296296, 85.14814814814815], [85.84444444444445, 85.86666666666667, 85.82037037037037], [86.29074074074074, 86.39259259259259, 86.43888888888888], [86.76666666666667, 86.71666666666667, 86.67222222222222], [87.18148148148148, 87.18333333333334, 87.17962962962963], [87.38703703703705, 87.56666666666668, 87.38333333333334], [87.80925925925926, 87.9, 87.75], [87.99444444444444, 88.08703703703704, 88.06296296296297], [88.35555555555555, 88.2611111111111, 88.40740740740742], [88.72037037037038, 88.57037037037037, 88.72592592592594], [88.80925925925925, 88.86111111111111, 88.88703703703705], [89.09814814814816, 89.07037037037037, 89.08703703703705], [89.3037037037037, 89.49444444444444, 89.28148148148148], [89.46296296296296, 89.66296296296296, 89.54444444444445], [89.72407407407408, 89.6037037037037, 89.75], [89.99074074074073, 89.78518518518518, 89.93148148148148], [90.32777777777777, 89.99074074074073, 90.07037037037037], [90.21851851851852, 90.36296296296297, 90.27592592592593], [90.45370370370371, 90.42592592592592, 90.51296296296296], [90.53333333333333, 90.67407407407407, 90.6962962962963], [90.85, 90.84444444444445, 90.76111111111112], [90.90925925925926, 91.05740740740741, 90.8425925925926], [91.06111111111112, 91.12777777777778, 91.11666666666667], [91.35740740740741, 91.23703703703704, 91.18518518518518], [91.48703703703703, 91.35925925925926, 91.42407407407407], [91.61851851851853, 91.44444444444444, 91.58148148148149]], [[16.13888888888889, 18.227777777777778, 16.522222222222222], [65.81851851851852, 66.42962962962963, 64.21666666666667], [75.80185185185185, 76.42962962962963, 76.06111111111112], [80.32407407407408, 80.63148148148149, 80.31481481481481], [82.5537037037037, 82.85925925925926, 82.78703703703704], [83.87592592592593, 84.09814814814814, 84.11481481481482], [84.89999999999999, 85.0537037037037, 85.12407407407407], [85.66666666666667, 85.92222222222222, 85.81296296296296], [86.39444444444445, 86.30740740740741, 86.41111111111111], [86.77962962962962, 86.87037037037038, 86.88148148148149], [87.09444444444443, 87.36851851851853, 87.43333333333332], [87.53333333333333, 87.79814814814814, 87.68703703703704], [87.81481481481481, 87.99259259259259, 87.98703703703704], [88.08703703703704, 88.48888888888888, 88.22777777777779], [88.47037037037036, 88.83148148148148, 88.56666666666668], [88.77777777777777, 88.97962962962963, 88.84814814814814], [89.04074074074074, 89.33703703703704, 88.97407407407407], [89.27592592592592, 89.41666666666667, 89.38148148148149], [89.54074074074074, 89.66481481481482, 89.49259259259259], [89.7537037037037, 89.90925925925926, 89.78518518518518], [89.83888888888889, 90.04629629629629, 89.72222222222223], [90.0037037037037, 90.3425925925926, 90.01851851851852], [90.24074074074075, 90.33333333333333, 90.35], [90.34814814814814, 90.5537037037037, 90.47222222222221], [90.52222222222223, 90.79074074074074, 90.62222222222222], [90.69444444444444, 90.90185185185186, 90.8037037037037], [90.67962962962963, 91.10000000000001, 90.97407407407407], [91.04629629629629, 91.2425925925926, 91.04444444444444], [91.24444444444444, 91.34629629629629, 91.18703703703703], [91.25555555555556, 91.49259259259259, 91.3537037037037]]]
    val_accuracies = [[[73.05, 73.15, 79.03333333333333], [81.93333333333334, 84.73333333333333, 80.86666666666666], [82.91666666666667, 81.86666666666666, 86.6], [85.13333333333334, 85.66666666666667, 84.89999999999999], [85.36666666666667, 85.1, 87.3], [87.56666666666668, 87.76666666666667, 86.76666666666667], [88.63333333333333, 87.06666666666666, 87.76666666666667], [87.36666666666667, 88.13333333333333, 85.18333333333334], [87.88333333333334, 87.18333333333334, 87.85], [84.56666666666666, 86.6, 88.88333333333334], [87.98333333333333, 88.63333333333333, 89.51666666666667], [88.64999999999999, 88.68333333333334, 87.43333333333332], [86.01666666666667, 88.1, 88.56666666666668], [89.28333333333333, 88.91666666666667, 88.86666666666667], [88.05, 87.53333333333333, 85.78333333333333], [88.31666666666666, 88.96666666666667, 89.1], [86.53333333333333, 88.31666666666666, 88.8], [89.08333333333334, 87.75, 90.11666666666667], [89.35, 86.7, 89.5], [89.23333333333333, 88.31666666666666, 88.41666666666667], [88.75, 89.1, 89.46666666666667], [87.9, 89.28333333333333, 89.38333333333334], [87.3, 89.63333333333333, 89.85], [89.01666666666667, 89.45, 88.76666666666667], [89.48333333333333, 88.56666666666668, 88.91666666666667], [89.8, 89.31666666666666, 89.85], [89.23333333333333, 89.45, 87.86666666666667], [88.28333333333333, 88.9, 89.11666666666666], [88.94999999999999, 90.11666666666667, 90.11666666666667], [89.11666666666666, 89.8, 90.21666666666667]], [[80.2, 79.31666666666666, 82.56666666666666], [82.31666666666668, 82.38333333333333, 84.16666666666667], [82.81666666666668, 84.18333333333334, 85.41666666666666], [85.05, 83.81666666666666, 84.89999999999999], [84.15, 85.3, 84.75], [85.53333333333333, 85.26666666666667, 86.03333333333333], [84.85000000000001, 86.25, 86.18333333333334], [85.96666666666667, 86.55000000000001, 84.53333333333333], [87.16666666666667, 86.78333333333333, 85.35000000000001], [86.66666666666667, 86.45, 86.35000000000001], [87.25, 86.88333333333334, 85.28333333333333], [85.88333333333334, 86.95, 87.33333333333333], [87.68333333333334, 87.56666666666668, 88.33333333333333], [87.06666666666666, 85.43333333333332, 85.66666666666667], [87.81666666666666, 86.08333333333333, 87.94999999999999], [87.94999999999999, 87.83333333333333, 88.58333333333334], [88.3, 87.7, 87.91666666666667], [88.08333333333334, 87.9, 87.68333333333334], [85.98333333333333, 87.98333333333333, 88.14999999999999], [87.85, 87.83333333333333, 88.36666666666667], [87.8, 87.41666666666667, 88.6], [88.23333333333333, 87.23333333333333, 87.63333333333333], [87.13333333333333, 87.85, 88.23333333333333], [86.4, 88.5, 88.5], [87.43333333333332, 88.63333333333333, 88.36666666666667], [88.5, 88.35, 88.43333333333334], [88.16666666666667, 89.0, 88.31666666666666], [88.05, 85.5, 88.75], [88.13333333333333, 86.98333333333333, 88.9], [88.53333333333333, 88.25, 88.26666666666667]], [[52.849999999999994, 44.35, 50.5], [73.01666666666667, 69.66666666666667, 72.05], [76.14999999999999, 80.2, 75.48333333333333], [81.78333333333333, 82.33333333333334, 82.98333333333333], [83.2, 83.8, 83.78333333333333], [83.01666666666667, 84.5, 85.03333333333333], [84.0, 82.88333333333333, 84.58333333333333], [86.08333333333333, 82.71666666666667, 85.85000000000001], [83.89999999999999, 82.83333333333334, 86.45], [86.75, 86.48333333333333, 86.51666666666667], [86.35000000000001, 86.7, 85.9], [87.0, 85.8, 87.76666666666667], [86.9, 84.7, 88.21666666666667], [87.06666666666666, 86.83333333333333, 87.28333333333333], [87.18333333333334, 87.63333333333333, 88.08333333333334], [88.38333333333334, 87.38333333333334, 87.58333333333333], [87.61666666666666, 88.26666666666667, 88.6], [87.68333333333334, 86.68333333333334, 88.41666666666667], [88.06666666666668, 87.28333333333333, 88.23333333333333], [86.48333333333333, 88.21666666666667, 88.25], [88.31666666666666, 87.7, 88.16666666666667], [87.85, 87.66666666666667, 88.44999999999999], [88.21666666666667, 87.26666666666667, 88.56666666666668], [88.55, 88.33333333333333, 88.7], [88.46666666666667, 87.76666666666667, 89.46666666666667], [88.56666666666668, 88.53333333333333, 88.44999999999999], [87.38333333333334, 88.25, 89.38333333333334], [88.86666666666667, 88.43333333333334, 88.03333333333333], [88.56666666666668, 88.93333333333334, 88.51666666666667], [86.45, 88.55, 86.83333333333333]]]

    plot_different_architecture_accuracies(train_accuracies, val_accuracies)


