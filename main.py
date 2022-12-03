from model_nn import *
import pandas as pd

data = pd.read_csv("data.csv")
X_train, X_val, y_train, y_val = preprocessing_split(data)

NN_train = NN(X_train, y_train, learning_rate = 0.1, momentum_rate = 0.1, n_neuron = 3)

NN_train.train()