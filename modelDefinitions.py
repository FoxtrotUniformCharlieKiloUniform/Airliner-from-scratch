import pandas
import torch
import numpy as np
import supporting_cast as sc
import math

#the goal of this is to do from scratch (aka, no torch F.abc or nn.xyz.)
#I'll use pytorch for the tensor handling

#support method unrelated to the actual model. Will return batch_size, a, b from a,b
#also turns dataframes into tensors
def fixup_data(data, batch_size, targetColumn, trainPercent):
    assert data.shape[0] % batch_size == 0
    
    #TODO: rewrite this first section to make it more efficient (get column index from input text, so you don't have to convert x and y to tensor seperately)
    x = data.drop(targetColumn, axis = 1)
    y = data[targetColumn]                      #split up data into target column, and remove that target column from the data
    
    
    x = torch.tensor(x.values)
    y = torch.tensor(y.values)

    x = torch.reshape(x, (batch_size, data.shape[1]-1, -1))
    y = torch.reshape(y, (batch_size, 1, -1)) 

    (x_train, x_test) = torch.split(x, round(trainPercent/100 * batch_size))   
    (y_train, y_test) = torch.split(y, round(trainPercent/100 * batch_size))
    return x_train, y_train, x_test, y_test



class layer():
    def __init__(self, in_size, out_size):
        self.weights = torch.rand(in_size, out_size)
        self.bias = torch.rand(1, out_size)
    
    def forward(self, inputs):
        self.weights = self.weights.to(dtype = torch.double)
        self.bias = self.bias.to(dtype = torch.double)
        self.inputs = torch.tensor(inputs[0]).to(dtype = torch.double)      #TODO: fix inputs growing to inf at some point 
        
        output = torch.matmul(self.inputs, self.weights) + self.bias
        return output
        
    def backward(self, output_error, lr):
        
        # Ensure the weights matrix and output_error align correctly for matrix multiplication
        output_error = output_error.to(dtype=torch.double).unsqueeze(1)

        # Calculate the error propagated back to the input
        input_error = torch.matmul(self.weights, output_error).squeeze()
        
        # Calculate the gradient for weights and biases
        a = self.inputs.unsqueeze(0).transpose(1,0)
        b = output_error.transpose(1,0)

        weights_error = torch.matmul(a,b)
        bias_error = output_error.squeeze(1)
        
        # Update the weights and biases
        self.weights = self.weights - lr * weights_error
        self.bias = self.bias - lr * bias_error
        
        return input_error


class nonLinearity(torch.nn.Module):
    def __init__(self, type):
        super().__init__()
        if type == "sigmoid":
            self.eq = lambda x: 1 / (1 + torch.exp(-x))
            self.deq = lambda x: self.eq(x) * (1 - self.eq(x))
        elif type == "relu":
            self.eq = lambda x: torch.maximum(x, torch.tensor(0.0))
            self.deq = lambda x: (x > 0).float()
        else:
            raise ValueError("Unrecognized input")

    def forward(self, input):
        self.input = input
        output = self.eq(input)
        return output

    def backward(self, output_error, lr):
        return output_error * self.deq(self.input)
    

#loss function
def mean_absolute_error(true, pred):
    return torch.mean(abs(true-pred))

def mean_absolute_derivative(true, pred):
#https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative - TODO validate this later
    pred = pred.squeeze(0)
    true = true.squeeze(0)

    outputVector = []
    for i in range(pred.shape[0]):
        if pred[i] > true[i]:
            outputVector.append(1)
        else:
            outputVector.append(-1)
    return torch.tensor(outputVector)

class FCN:
    def __init__(self):
        self.layersList = []
        self.loss = 0
        self.dLoss = 0

    def addLayer(self, layer):
        self.layersList.append(layer)
        pass
    
    def forward(self, input):
        layerInput = input
        for layerNum, layer in enumerate(self.layersList):
            layerInput = layer.forward(layerInput)

        output = layerInput

        return output
    
    def backward(self, loss_grad, lr):
        layer_grad = loss_grad
        
        for layer in reversed(self.layersList):
            layer_grad = layer.backward(layer_grad, lr)


def predictAirlinePriceFromIndex(network, df, idx):
    row = df.loc[idx]
    row = row.drop("fare_low")
    row = torch.tensor(row).unsqueeze(0)
    print(row)
    price = network.forward(row)
    return price


            

