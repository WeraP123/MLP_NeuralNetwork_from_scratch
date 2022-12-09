import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def preprocessing_split(data, split):
    data = (data - data.min()) / (data.max() - data.min())
    x = data[['x','y']]
    y = data[['vel_x','vel_y']]
    x = x.to_numpy()
    y = y.to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size= split, random_state=12)
    return X_train, X_val, y_train, y_val


class NN:
    def __init__(self, inputs, outputs, learning_rate, momentum_rate, n_neuron):

        self.inputs = inputs
        self.outputs = outputs

        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.n_neuron = n_neuron

        # jak dac wag??
        self.W2 = np.random.rand(2, n_neuron)
        self.W1 = np.random.rand(2 , n_neuron)
        self.iteration = 0
        self.bias1 = np.random.randn(1, n_neuron)
        self.bias2 = np.random.randn(1, 1)
        self.temp_delta = []
        self.temp_delta_hid = []

    # W2 = np.random.rand(2, n_neuron)

    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v * self.learning_rate))

    def derivative_sigmoid(self, v):
        return v * (1 - v)

    def feedforward(self):
        self.Z = self.sigmoid(np.dot(self.x, self.W1) + self.bias1)
        self.pred_output = np.dot(self.Z, self.W2.T) + self.bias2
        return self.pred_output

    def RMSE(self, error):
        mse = np.square(error).mean()
        rmse = math.sqrt(mse)
        return rmse

    def train(self):
        epoch_counter = 0
        rmse_err = []
        errors = []

        while epoch_counter < 80:
            for i in range(self.inputs.shape[0]):

                self.x = self.inputs[i]
                self.y = self.outputs[i].T
                self.feedforward()
                self.backpropagation()
                errors.append(self.error_signal)
                self.iteration = self.iteration + 1


            epoch_counter = epoch_counter + 1
            mse = np.square(errors).mean()
            rmse = math.sqrt(mse)
            rmse_err.append(rmse)

        np.savetxt('W2.txt', self.W2)
        np.savetxt('W1.txt', self.W1)
        plt.plot(range(0,epoch_counter), rmse_err)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()


        print("Finished with error", rmse_err[-1])


    def backpropagation(self):

        self.error_signal = self.y - self.pred_output

        delta_weight = self.learning_rate * np.dot(self.Z.T, self.error_signal)
        h_grad = self.learning_rate * (self.derivative_sigmoid(self.Z) * np.dot(self.error_signal, self.W2))
        delta_hidden = np.dot(np.reshape(self.x, (2,1)), h_grad)

        self.temp_delta.append(delta_weight.T)
        self.temp_delta_hid.append(delta_hidden)


        self.bias1 = self.bias1 + sum(delta_hidden)
        self.bias2 = self.bias2 + sum(delta_weight)

        if self.iteration == 0:
            self.W2 = np.reshape(delta_weight, (2, self.n_neuron)) + self.W2
            self.W1 = np.reshape(delta_hidden, (2, self.n_neuron)) + self.W1
        else:
            self.W2 = np.reshape(delta_weight, (2, self.n_neuron)) + self.W2 + self.momentum_rate * np.array(self.temp_delta[-1])
            self.W1 = np.reshape(delta_hidden, (2, self.n_neuron)) + self.W1 + self.momentum_rate * np.array(self.temp_delta_hid[-1])

    def predict(self, new_input):
        pred = self.feedforward()
        return pred


