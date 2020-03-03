import torch.nn as nn


def _outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * (padding)) / stride) + 1
    return (output)


class GraphNet(nn.Module):
    def __init__(self, input_size, num_neurons=500):
        super(GraphNet, self).__init__()

        # convolutional layers w/
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=20,
        #                        kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv2 = nn.Conv2d(input_size, input_size // 2,
        #                        kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(input_size, num_neurons)

        # FC layers with 500 fully-connected neurons at the end
        self.fc2 = nn.Linear(num_neurons, 64)

        # FC final output is 2-class
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Use sigmoid to convert the output into range (0,1)

    def forward(self, x):
        # compute activation from first convolution
        # (N, N) to (20, N, N)
        # x = F.relu(self.fc1(x))
        # pool and downsize by a factor of 2
        # (20, N/2, N/2)
        # x = self.pool1(x)
        # reshape the data and flatten the layer
        # x = x.view(-1,)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x).view(-1)
        return x


class My_Net(nn.Module):
    def __init__(self, input_size, num_neurons):
        super(My_Net, self).__init__()
        # feed forward layers
        self.layer_1 = nn.Linear(input_size, num_neurons)
        self.layer_2 = nn.Linear(num_neurons, num_neurons)
        self.layer_3 = nn.Linear(num_neurons, 1)

        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Use sigmoid to convert the output into range (0,1)

    def forward(self, input_data):
        out = self.layer_1(input_data)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.layer_3(out)
        out = self.sigmoid(out).view(-1)
        return out
