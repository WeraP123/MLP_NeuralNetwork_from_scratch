import numpy as np
from model_nn import NN
import pickle

class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.net = NN(None, None, learning_rate=0.1, momentum_rate=0.1, n_neuron=8)
        self.weights = np.load("./weights.npy")
        self.net.W1 = self.weights[0]
        self.net.W2 = self.weights[1]

        with open("normalization_values.pkl", "rb") as f:
            self.normalization_values = pickle.load(f)

    def predict(self, input_row):
        input_data = np.fromstring(input_row, dtype=float, sep=',')

        x, y = input_data

        x = x * (self.normalization_values.get("x_max") - self.normalization_values.get("x_min")) + self.normalization_values.get("x_min")
        y = y * (self.normalization_values.get("y_max") - self.normalization_values.get("y_min")) + self.normalization_values.get("y_min")

        y_pred = self.net.predict(np.array([x, y]))[0]

        vel_x, vel_y = y_pred

        vel_x = vel_x * (self.normalization_values.get("vel_x_max") - self.normalization_values.get("vel_x_min")) + self.normalization_values.get("vel_x_min")
        vel_y = vel_y * (self.normalization_values.get("vel_y_max") - self.normalization_values.get("vel_y_min")) + self.normalization_values.get("vel_y_min")

        return np.array([vel_x, vel_y])
