import sys
import os
import re
import math
import logging
import argparse
import math
import numpy as np 


def read_data(file_name):
    # Reads .dat files
    f = open(file_name, 'r')

    data = []
    f.readline()
    for instance in f.readlines():
        if not re.search('\t', instance): continue
        data += [list(map(int, instance.strip().split('\t') ))]
    return data


def dot_product(array1, array2):
    # Returns the dot product between array1 and array2
    #####################
    # YOUR CODE GOES HERE
    #####################
    print(array1)
    print(array2)
    print(np.dot(array1, array2))
    result = sum([i*j for (i, j) in zip(array1, array2)])
    print(result)
    return result


def sigmoid(x):
    # Returns sigmoid of x
    #####################
    # YOUR CODE GOES HERE
    #####################
    result = 1/(1+math.exp( -x ))
    return result


def predict(weights, instance):
    #####################
    # YOUR CODE GOES HERE
    #####################
    output = sigmoid(dot_product(weights, instance))
    if output > 0.5:
        return 1
    return 0


def train_perceptron(instances, lr, epochs):
    # Train (calculates weights) for a sigmoid perceptron
    weights = [0] * (len(instances[0])-1)

    #####################
    # YOUR CODE GOES HERE
    #####################
    return weights


def get_accuracy(weights, instances):
    ###########################
    # DO NOT CHANGE THIS METHOD
    ###########################
    # Predict instances and return accuracy
    error = 0
    for instance in instances:
        prediction = predict(weights, instance)
        error += abs(instance[-1] - prediction)
   
    accuracy = float(len(instances)-error) / len(instances)
    return accuracy * 100


def main(file_tr, file_te, lr, epochs):
    ###########################
    # DO NOT CHANGE THIS METHOD
    ###########################
    instances_tr = read_data(file_tr)
    instances_te = read_data(file_te)

    # Training: calculate weights
    weights = train_perceptron(instances_tr, lr, epochs)
    # Testing: calculate accuracy in the test set
    accuracy = get_accuracy(weights, instances_te)
    print(f"Accuracy on test set ({len(instances_te)} instances): {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR", help="Path to train file with POS annotations")
    parser.add_argument("PATH_TE", help="Path to test file (POS tags only used for evaluation)")
    parser.add_argument("lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("epochs", type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    print(' '.join(map(str, [args.PATH_TR, args.PATH_TE, args.lr, args.epochs])))

    main(args.PATH_TR, args.PATH_TE, args.lr, args.epochs)
