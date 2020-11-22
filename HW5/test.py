'''
# Import required libraries:
import numpy as np
# Define input features:
input_features = np.array([[0,0],[0,1],[1,0],[1,1]])


target_output = np.array([[0,1,1,1]])
target_output = target_output.reshape(4,1)

weights = np.array([[0.1],[0.2]])

bias = 0.3
# Learning Rate:
lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# Main logic for neural network:
 # Running our code 10000 times:
for epoch in range(10000):
    inputs = input_features
    #Feedforward input:
    in_o = np.dot(inputs, weights) + bias 
    # print(inputs)
    # print(weights)
    # print(np.dot(inputs, weights))
    # print(in_o)
    # Feedforward output:
    out_o = sigmoid(in_o) 
    # print(out_o)
    #Backpropogation 
    #Calculating error
    error = out_o - target_output
    # print(error)
    
    #Going with the formula:
    x = error.sum()
    # print(x)
    
    #Calculating derivative:
    derror_douto = error
    douto_dino = sigmoid_der(out_o)
    # print(douto_dino)
    
    #Multiplying individual derivatives:
    deriv = derror_douto * douto_dino 
    #Multiplying with the 3rd individual derivative:
    #Finding the transpose of input_features:
    inputs = input_features.T
    deriv_final = np.dot(inputs,deriv)
    
    #Updating the weights values:
    weights -= lr * deriv_final 
    #Updating the bias weight value:
    for i in deriv:
        bias -= lr * i 
    
#Check the final values for weight and bias
print (weights)
print (bias) 

#Taking inputs:
single_point = np.array([1,0]) 
#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2) 

#Taking inputs:
single_point = np.array([1,1]) 
#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2) 

#Taking inputs:
single_point = np.array([0,0]) 
#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2)
'''

# Import required libraries :
import numpy as np
# Define input features :
input_features = np.array([[0,0],[0,1],[1,0],[1,1]])

# Define target output :
target_output = np.array([[0,1,1,1]])
# Reshaping our target output into vector :
target_output = target_output.reshape(4,1)
weights = np.array([[0.0],[0.0]])

# Define learning rate :
lr = 0.01
# Sigmoid function :
def sigmoid(x):
    return 1/(1+np.exp(-x))# Derivative of sigmoid function :
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))
# Main logic for neural network :
# Running our code 10000 times :
for epoch in range(10):
    inputs = input_features
    #Feedforward input :
    pred_in = np.dot(inputs, weights)
  
    #Feedforward output :
    pred_out = sigmoid(pred_in)
  
    #Backpropogation 
    #Calculating error
    error = pred_out - target_output
    x = error.sum()

    #Going with the formula :
    print(x)

    #Calculating derivative :
    dcost_dpred = error
    dpred_dz = sigmoid_der(pred_out)

    #Multiplying individual derivatives :
    z_delta = dcost_dpred * dpred_dz
    #Multiplying with the 3rd individual derivative :
    inputs = input_features.T
    weights -= lr * np.dot(inputs, z_delta)

    # print(weights)
 

#Taking inputs :
single_point = np.array([1,0])
#1st step :
result1 = np.dot(single_point, weights)
#2nd step :
result2 = sigmoid(result1)
#Print final result
print(result2)

#Taking inputs :
single_point = np.array([0,0])
#1st step :
result1 = np.dot(single_point, weights)
#2nd step :
result2 = sigmoid(result1)
#Print final result
print(result2)

#Taking inputs :
single_point = np.array([1,1])
#1st step :
result1 = np.dot(single_point, weights)
#2nd step :
result2 = sigmoid(result1)
#Print final result
print(result2)
