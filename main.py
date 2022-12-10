from model_nn import *
import pandas as pd

# reading in data
data = pd.read_csv("data.csv")

# split and scale data
X_train, X_val, y_train, y_val = preprocessing_split(data, 0.33)

# initialise the model
NN_train = NN(X_train, y_train, learning_rate=0.03, momentum_rate=0.1, n_neuron=8, epochs=10000)

# train the model
NN_train.train()

# get a prediction
pred_y = NN_train.predict(X_val)

# evaluate
error = y_val - pred_y
mse = np.square(error).mean()
rmse_val = math.sqrt(mse)
print("Validation score: ", rmse_val)