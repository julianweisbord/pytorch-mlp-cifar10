'''
Created on May 12th, 2018
author: Julian Weisbord
sources: https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb
         https://www.cs.toronto.edu/~kriz/cifar.html
description: This is a Multilayer Perceptron image classifier trained on the
             CIFAR-10 data set.
'''

import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Constants
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 3
EPOCHS = 300
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1, 1]
KEEP_RATES = [.5, .6, .7, .8, .9]

BATCH_SIZE = 32
BATCH_IMAGE_COUNT = 10000
TRAIN_BATCHS = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
TEST_BATCHES = ["data_batch_5"]
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
N_CLASSES = len(CLASSES)
PLOT = False



class Net(torch.nn.Module):
    def __init__(self, n_hidden_nodes, n_hidden_layers, activation, keep_rate=0):
        super(Net, self).__init__()
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        if not keep_rate:
            keep_rate = 0.5
        self.keep_rate = keep_rate
        # Set up perceptron layers and add dropout
        self.fc1 = torch.nn.Linear(IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS,
                                   n_hidden_nodes)
        self.fc1_drop = torch.nn.Dropout(1 - keep_rate)
        if n_hidden_layers == 2:
            self.fc2 = torch.nn.Linear(n_hidden_nodes,
                                       n_hidden_nodes)
            self.fc2_drop = torch.nn.Dropout(1 - keep_rate)

        self.out = torch.nn.Linear(n_hidden_nodes, N_CLASSES)

    def forward(self, x):
        x = x.view(-1, IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS)
        if self.activation == "sigmoid":
            sigmoid = torch.nn.Sigmoid()
            x = sigmoid(self.fc1(x))
        elif self.activation == "relu":
            x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc1_drop(x)
        if self.n_hidden_layers == 2:
            if self.activation == "sigmoid":
                x = sigmoid(self.fc2(x))
            elif self.activation == "relu":
                x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc2_drop(x)
        return torch.nn.functional.log_softmax(self.out(x))


def train(epoch, model, train_loader, optimizer, log_interval=100, cuda=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def validate(loss_vector, accuracy_vector, model, validation_loader, cuda=None):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += torch.nn.functional.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))

def main():
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=1, pin_memory=False)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    validation_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=1, pin_memory=False)

    hidden_nodes = 100
    layers = 1
    for i in range(1, len(LEARNING_RATES) + 1):
        model = Net(hidden_nodes, layers, "sigmoid")
        if cuda:
            model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATES[i-1])

        loss_vector = []
        acc_vector = []
        for epoch in range(1, EPOCHS + 1):
            train(epoch, model, train_loader, optimizer, cuda=cuda)
            validate(loss_vector, acc_vector, model, validation_loader, cuda=cuda)
            if epoch == 10:
                break

        # Plot train loss and validation accuracy vs epochs for each learning rate
        if PLOT:
            epochs = [i for i in range(1, 11)]
            plt.plot(epochs, acc_vector)
            plt.plot(epochs, loss_vector)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.show()
    # Repeat using RELU for activation

    hidden_nodes = 100
    layers = 1
    for i in range(1, len(LEARNING_RATES) + 1):
        model = Net(hidden_nodes, layers, "relu")
        if cuda:
            model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATES[2])

        loss_vector = []
        acc_vector = []
        for epoch in range(1, EPOCHS + 1):
            train(epoch, model, train_loader, optimizer, cuda=cuda)
            validate(loss_vector, acc_vector, model, validation_loader, cuda=cuda)
            if epoch == 10:
                break
        # Plot train loss and validation accuracy vs epochs for each learning rate
        if PLOT:
            epochs = [i for i in range(1, 11)]
            plt.plot(epochs, acc_vector)
            plt.plot(epochs, loss_vector)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.show()

if __name__ == '__main__':
    main()
