
### Abstract
Implement the gradient descent algorithm that we discussed in class to train a sigmoid unit.

This code only handle binary classification tasks (each instance will have class 0 or 1).

In addition, assume that all attributes have binary values (either 0 or 1).

When applying the trained sigmoid unit to a test instance, use 0.5 as the classification threshold (i.e., classify the instance as 1 if the unit outputs a value that is greater or equal than 0.5, otherwise classify the instance as 0). Initialize all the weights to 0.

### Usage:
$ python perceptron.py data/train.dat data/test.dat 0.01 10