# Homework 2

Adam Li
03/03/2020

## Introduction
We generate n x n graphs with n = 4, 5, 6 and determine their largest stable set. We then train a simple neural network model 
on a training subset of data and report accuracy on a held-out test dataset.

## Network Structure:
We used a fully-connected MLP neural network on all datasets with the ADAM optimizer. In terms of network architecture, we keep it as a 3-layer fully connected MLP with 20 hidden units
in each layer. We use a nonlinearity of RELU. Here is an example of our neural network structure for graphs of size 4.
 
    GraphNet(
      (fc1): Linear(in_features=6, out_features=20, bias=True)
      (fc2): Linear(in_features=20, out_features=20, bias=True)
      (fc3): Linear(in_features=20, out_features=3, bias=True)
      (relu): ReLU()
      (sigmoid): Softmax(dim=1)
    )
 
## Training:

We used fixed hyperparameters with:
    
    * batch_size = x
    * epochs = 50
    * learning_rate = 5e-3
    
We vary the hyperparameters from 1, 5, and 10 based on the size of the dataset we are working with. Stable 4 and 5 and 6
each have 2 to the power of n squared choose 2 possible graph configurations.

|                     | Stable4       | Stable4      | Stable5       | Stable5      | Stable6       | Stable6      |
|---------------------|---------------|--------------|---------------|--------------|---------------|--------------|
| Activation Function | End Train Acc | End Test Acc | End Train Acc | End Test Acc | End Train Acc | End Test Acc |
| RELU                | 97.727        | 89.476       | 88.826        | 85.389       | 80.398        | 80.246             |

## Discussion
As expected, we saw lower performance as we increased n in our graph size. This is due to the combinatorial nature of this problem.
Finding the largest stable set of a graph is NP-hard, and so the number of possible combinations that give different outputs grows, as
well as the possible y labels. 

With graphs, we could have made use of graph convolutional neural networks, which performs convolutions over graph space (i.e. 
permutations), and learns weights that operate on a graph. Since we vectorized the graph, we would have to convert it into graph
form. 