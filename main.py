import pandas
import torch
import numpy as np
import supporting_cast as sc
import math
import modelDefinitions


#in this dataframe, we have a couple variables of interest, all of which are regressive.
#so we have tabular, non timeseries style data. 

#For now, this is just going to be a shrimple X layer fully connected deep neural network. To define a layer, need a bunch of nodes (obviously)
#as per https://hastie.su.domains/Papers/ESLII.pdf

#variable definitions
batch_size = 64
epochs = 2
trainPercent = 90       #percent of data we will be using for training
learning_rate = 0.01
prediction_idx = 2616

number_of_training_examples = round(trainPercent/100 * batch_size)

#final data preprocessing before model
df = sc.airlineSimplified
df.drop(["airportid_1", "airportid_2"], axis = 1, inplace = True)

x_train, y_train, x_test, y_test = modelDefinitions.fixup_data(df, batch_size, "fare_low",trainPercent)

outputSize = y_train.shape[-1]

#defining model
network = modelDefinitions.FCN()
fc1 = modelDefinitions.layer(152 * 7, 250)
fc2 = modelDefinitions.layer(250, 200)
#nonlinearLayer = modelDefinitions.nonLinearity("relu")
fc3 = modelDefinitions.layer(200, 152)

network.addLayer(fc1)
network.addLayer(fc2)
#network.addLayer(nonlinearLayer)
network.addLayer(fc3)

for epoch in range(epochs):
    print(f"Epoch {epoch+1} / {epochs}")
    for batch in range(number_of_training_examples - 1):
        x_train_i = x_train[batch].flatten().unsqueeze(0)     #remove batch, reduce size to "a" column vectors        (where a is the amount of data we choose to pass in, like market share or passengers or whatever)
        y_train_i = y_train[batch]#.unsqueeze(0)     #remove batch, reduce size to one column vector

        #model actuation
        outputs = network.forward(x_train_i)
        loss = modelDefinitions.mean_absolute_error(y_train_i, outputs)
        dLoss = modelDefinitions.mean_absolute_derivative(y_train_i, outputs)

        grad = network.backward(dLoss, learning_rate)


#pricePrediction = modelDefinitions.predictAirlinePriceFromIndex(network = network, df = df, idx = prediction_idx)

#print(f"Predicted price for index of {prediction_idx} is {pricePrediction}")