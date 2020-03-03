import time
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from dldo.training.models import GraphNet

# set random seeds for reproducible results
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def createLossAndOptimizer(net, learning_rate=0.001):
    # Loss function
    loss = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    return (loss, optimizer)

#initialize the network using Xavier initialization.
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

def train_eval(verbose = 1):
    correct = 0
    total = 0
    loss_sum = 0
    num_batches = 0
    for inputs, labels in train_loader:
        outputs = net(inputs)
        predicted = outputs.data>0.5
        total += labels.size(0)
        correct += (predicted.int() == labels.int()).sum()
        loss_sum  += loss(outputs,labels).item()
        num_batches += 1

    if verbose:
        print('Train accuracy: %f %%' % (100 * correct.item() / total))
    return loss_sum/num_batches, correct.item() / total

def test_eval(verbose = 1):
    correct = 0
    total = 0
    loss_sum = 0
    num_batches = 0
    for inputs, labels in test_loader:
        outputs = net(inputs)
        predicted = outputs.data>0.5
        total += labels.size(0)
        correct += (predicted.int() == labels.int()).sum()
        loss_sum  += loss(outputs,labels).item()
        num_batches += 1

    if verbose:
        print('Test accuracy: %f %%' % (100 * correct.item() / total))
    return loss_sum/num_batches, correct.item() / total

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

    # Loop for n_epochs
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs, labels = data

            # Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)

            print(inputs.shape, labels.shape)
            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # evaluate training loss and accuracy
            l_temp, acc_temp = train_eval()
            train_loss_store.append(l_temp)
            train_acc_store.append(acc_temp)

            # evaluate testing loss and accuracy
            l_temp, acc_temp = test_eval()
            test_loss_store.append(l_temp)
            test_acc_store.append(acc_temp)

            # Print statistics
            running_loss += l_temp
            total_train_loss += l_temp

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in test_loader:
            # Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels).item()
            total_val_loss += val_loss_size

        print("Validation loss = {:.2f}".format(total_val_loss / len(test_loader)))
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


if __name__ == '__main__':
    # directory file path
    data_dir = Path("/Users/adam2392/Documents/dldo/data/raw/hw2")
    data_fname = "stable4.txt"
    data_fpath = Path(data_dir / data_fname)

    # load in the file path
    X = []
    y = []
    with open(data_fpath, 'r') as fin:
        for i in range(6):
            fin.readline()
        for line in fin:
            line = list(map(int, line.strip().split(' ')))
            X.append(line[:-1])
            y.append(line[-1])
    X = np.array(X)
    y = np.array(y)

    # initialize
    input_size = len(line[:-1])
    # define neural network
    net = GraphNet(input_size=input_size)
    print(net)

    # initialize loss function and nn optimizer
    loss, optimizer = createLossAndOptimizer(net)

    # initialize training parameters
    shuffle_dataset = True
    learning_rate = 0.001
    n_epochs = 10
    batch_size = 4
    validation_split = .1
    test_split = .2

    # create training dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle_dataset, test_size=0.2)
    Xtrain = torch.Tensor(X_train)
    Xtest = torch.Tensor(X_test)
    ytrain = torch.Tensor(y_train)
    ytest = torch.Tensor(y_test)

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