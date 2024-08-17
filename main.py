import pandas
import torch
import numpy as np
import supporting_cast as sc
import math
import modelDefinitions

# Variable definitions
batch_size = 64  # Thiais now flexible
epochs = 10
trainPercent = 90
learning_rate = 0.0001
prediction_idx = 2616

# Data preprocessing
df = sc.airlineSimplified
df.drop(["airportid_1", "airportid_2"], axis=1, inplace=True)

x_train, y_train, x_test, y_test = modelDefinitions.fixup_data(df, batch_size, "fare_low", trainPercent)

input_size = x_train.shape[1]  # Determine the input size dynamically
output_size = 1  # Assuming predicting a single value

# Defining model
network = modelDefinitions.FCN()
fc1 = modelDefinitions.layer(input_size, 250)
#midNL = modelDefinitions.nonLinearity("relu")
fc2 = modelDefinitions.layer(250, 200)
#midNL = modelDefinitions.nonLinearity("relu")
fc3 = modelDefinitions.layer(200, output_size)
#midSig = modelDefinitions.nonLinearity("sigmoid")

network.addLayer(fc1)
#network.addLayer(midNL)
network.addLayer(fc2)
network.addLayer(fc3)

# Training loop
for epoch in range(epochs):
    num_batches = x_train.shape[0] // batch_size
    
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        outputs = network.forward(x_batch)
        loss = modelDefinitions.mean_absolute_error(y_batch, outputs)
        dLoss = modelDefinitions.mean_absolute_derivative(y_batch, outputs)

        grad = network.backward(dLoss, learning_rate)

    print(f"Epoch {epoch+1} / {epochs}")

pricePrediction = modelDefinitions.predictAirlinePriceFromIndex(network = network, df = df, idx = prediction_idx)

print(f"Predicted price for index of {prediction_idx} is {pricePrediction}")
