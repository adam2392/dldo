import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

from dldo.training.models import GraphNet

# set random seeds for reproducible results
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def createLossAndOptimizer(net, learning_rate=0.001):
    # Loss function
    # loss = nn.BCELoss()
    loss = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    return (loss, optimizer)


# initialize the network using Xavier initialization.
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


def train_eval(verbose=1):
    correct = 0
    total = 0
    loss_sum = 0
    num_batches = 0
    for inputs, labels in train_loader:
        # get nn output and predictions
        outputs = net(inputs)
        predicted = torch.argmax(outputs, dim=1)

        # determine loss and accuracy
        total += labels.size(0)
        correct += (predicted.int() == labels.int()).sum()
        loss_sum += loss(outputs, labels).item()
        num_batches += 1

    if verbose:
        print('Train accuracy: %f %%' % (100 * correct.item() / total))
    return loss_sum / num_batches, correct.item() / total


def test_eval(verbose=1):
    correct = 0
    total = 0
    loss_sum = 0
    num_batches = 0
    for inputs, labels in test_loader:
        # get nn output and predictions
        outputs = net(inputs)
        predicted = torch.argmax(outputs, dim=1)

        # determine loss and accuracy
        total += labels.size(0)
        correct += (predicted.int() == labels.int()).sum()
        loss_sum += loss(outputs, labels).item()
        num_batches += 1

    if verbose:
        print('Test accuracy: %f %%' % (100 * correct.item() / total))
    return loss_sum / num_batches, correct.item() / total


def train_net(net, train_loader, test_loader, batch_size, n_epochs, learning_rate):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # define metrics to look at with loss and accuracy
    train_loss_store = []
    train_acc_store = []
    test_loss_store = []
    test_acc_store = []

    # Get training data
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()

    print_every = n_batches // 5

    # Loop for n_epochs
    for epoch in range(n_epochs):
        verbose = False
        if (epoch == 0) or ((epoch + 1) % print_every) == 0:
            verbose = True

        for i, (inputs, labels) in enumerate(train_loader, 0):
            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

        if verbose:
            print(f"Epoch {epoch + 1}:")

        # evaluate training loss and accuracy
        l_temp, acc_temp = train_eval(verbose=verbose)
        train_loss_store.append(l_temp)
        train_acc_store.append(acc_temp)

        # evaluate testing loss and accuracy
        l_temp, acc_temp = test_eval(verbose=verbose)
        test_loss_store.append(l_temp)
        test_acc_store.append(acc_temp)

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    # evaluate training loss and accuracy
    l_temp, acc_temp = train_eval(verbose=True)

    # evaluate testing loss and accuracy
    l_temp, acc_temp = test_eval(verbose=True)


if __name__ == '__main__':
    # directory file path
    data_dir = Path("/Users/adam2392/Documents/dldo/data/raw/hw2")
    data_fname = "stable6.txt"
    data_fpath = Path(data_dir / data_fname)

    # read in the data
    # X, y = read_from_txt(data_fpath)
    # Input the data and split into features and labels
    data = np.genfromtxt(data_fpath, dtype=np.int64, skip_header=7)
    X = data[:, :-1]
    y = data[:, -1]
    y -= 1  # index at 0

    # initialize
    input_size = X.shape[1]
    n_classes = len(np.unique(y))

    # define neural network
    num_neurons = 20
    net = GraphNet(input_size=input_size,
                   n_classes=n_classes,
                   num_neurons=num_neurons)
    print(net)

    # initialize loss function and nn optimizer
    loss, optimizer = createLossAndOptimizer(net)

    # initialize training parameters
    shuffle_dataset = True
    learning_rate = 5e-3
    n_epochs = 50
    batch_size = 10
    test_split = .3

    # create training dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        shuffle=shuffle_dataset,
                                                        test_size=test_split)
    Xtrain = torch.Tensor(X_train)
    Xtest = torch.Tensor(X_test)
    ytrain = torch.LongTensor(y_train)
    ytest = torch.LongTensor(y_test)

    # define pytorch Dataset
    train_set = torch.utils.data.TensorDataset(Xtrain, ytrain)
    test_set = torch.utils.data.TensorDataset(Xtest, ytest)

    # # Test and validation loaders have constant batch sizes, so we can define them directly
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              # sampler=test_sampler,
                                              num_workers=2)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               # sampler=train_sampler,
                                               num_workers=2)
    '''  BEGIN TRAINING PROCEDURE  '''
    # initialize nn weights
    net.apply(weights_init)

    # perform training
    train_net(net, train_loader, test_loader, batch_size, n_epochs, learning_rate)
