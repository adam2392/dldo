import torch.nn as nn


def _outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * (padding)) / stride) + 1
    return (output)


class GraphNet(nn.Module):
    def __init__(self, input_size, n_classes, num_neurons=32):
        super(GraphNet, self).__init__()

        # FC layers
        self.fc1 = nn.Linear(input_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, n_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Softmax(dim=1)  # Use sigmoid to convert the output into range (0,1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.relu(x)
        x = self.sigmoid(x)
        return x

