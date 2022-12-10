import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle


def preprocessing_split(data, split):

    """
    Scales data using the min-max method to the range (0, 1)
    Partitions the data
    :param data:
    :param split: defines the proportion of data split into val and train
    :return: Return normalised and split data into train and validation data sets
    """


    normalization_values = {
        "x_min": data.min().x,
        "x_max": data.max().x,

        "y_max": data.max().y,
        "y_min": data.min().y,


        "vel_x_min": data.min().vel_x,
        "vel_x_max": data.max().vel_x,

        "vel_y_min": data.min().vel_y,
        "vel_y_max": data.max().vel_y,
    }

    with open("normalization_values.pkl", "wb") as f:
        pickle.dump(normalization_values, f)

    data = (data - data.min()) / (data.max() - data.min())

    x = data[['x', 'y']]
    y = data[['vel_x', 'vel_y']]
    x = x.to_numpy()
    y = y.to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=split)
    return X_train, X_val, y_train, y_val


class NN:
    def __init__(self, inputs, outputs, learning_rate, momentum_rate, n_neuron, epochs=30):

        self.inputs = inputs
        self.outputs = outputs

        self.epochs = epochs

        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.n_neuron = n_neuron

        self.W2 = np.random.rand(2, n_neuron)*10
        self.W1 = np.random.rand(2, n_neuron)*10
        self.iteration = 0
        self.bias1 = np.random.randn(1, n_neuron)
        self.bias2 = np.random.randn(1, 1)
        self.temp_delta = []
        self.temp_delta_hid = []


    def sigmoid(self, v):
        """
        Getting activation value using the sigmoid function

        :param v: value of the given neuron
        :return: activation value of the given neuron
        """
        return 1 / (1 + np.exp(-v * self.learning_rate))

    def derivative_sigmoid(self, v):

        """
        :param v: value of the given neuron
        :return: derivative of the acivation fucntion sigmoid
        """

        return v * (1 - v)

    def feedforward(self, x):

        """
        Feedforward process:
        Matrix Z with activation values is defined for neurons in the hidden layer
        Values of the output layer are being calculated by passing activation matrix Z
        :param x: input row
        :return: returns the predicted value
        """
        self.Z = self.sigmoid(np.dot(x, self.W1) + self.bias1)
        self.pred_output = np.dot(self.Z, self.W2.T) + self.bias2
        return self.pred_output

    def RMSE(self, error):
        """
        RMSE - root mean squared error

        :param error: array of signal errors ( y - y_hat)
        :return: returns RMSE
        """
        mse = np.square(error).mean()
        rmse = math.sqrt(mse)
        return rmse

    def train(self):
        # initialising the epoch counter, array to store the signal errors per step and the aray to store rmse per epoch
        epoch_counter = 0
        rmse_err = []
        errors = []
        try:
            for epoch_counter in tqdm(range(self.epochs)):
                # feeding in data row by row
                for i in range(self.inputs.shape[0]):
                    self.y = self.outputs[i].T
                    self.feedforward(self.inputs[i])
                    self.backpropagation(self.inputs[i])
                    errors.append(self.error_signal)
                    self.iteration = self.iteration + 1

                epoch_counter = epoch_counter + 1

                # calculating rmse per epoch and storing it
                mse = np.square(errors).mean()
                rmse = math.sqrt(mse)
                rmse_err.append(rmse)

                # saving weights
                weights = np.array([self.W1, self.W2])
                np.save("weights.npy", weights)

                # printing the rmse per epoch
                print(rmse)

        except KeyboardInterrupt as e:
            print("Training is being stopped.")
        finally:
            print("Weights are being saved.")
            # saving final weights
            weights = np.array([self.W1, self.W2])
            np.save("weights.npy", weights)

        # plotting the error per epoch
        plt.plot(range(0, epoch_counter), rmse_err)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()

        # printing final error
        print("Finished with error", rmse_err[-1])

    def backpropagation(self, x):

        # calcutating error signal
        self.error_signal = self.y - self.pred_output

        # calculating delta values and hidden gradient value
        delta_weight = self.learning_rate * np.dot(self.Z.T, self.error_signal)
        h_grad = self.learning_rate * (self.derivative_sigmoid(self.Z) * np.dot(self.error_signal, self.W2))
        delta_hidden = np.dot(np.reshape(x, (2, 1)), h_grad)

        # saving the gradient for the purpose of adding momentum to the gradient search
        self.temp_delta.append(delta_weight.T)
        self.temp_delta_hid.append(delta_hidden)

        # updating bias
        self.bias1 = self.bias1 + sum(delta_hidden)
        self.bias2 = self.bias2 + sum(delta_weight)

        # updating weights
        if self.iteration == 0:
            self.W2 = np.reshape(delta_weight, (2, self.n_neuron)) + self.W2
            self.W1 = np.reshape(delta_hidden, (2, self.n_neuron)) + self.W1
        else:
            self.W2 = np.reshape(delta_weight, (2, self.n_neuron)) + self.W2 + self.momentum_rate * np.array(
                self.temp_delta[-1])
            self.W1 = np.reshape(delta_hidden, (2, self.n_neuron)) + self.W1 + self.momentum_rate * np.array(
                self.temp_delta_hid[-1])

    def predict(self, new_input):
        pred = self.feedforward(new_input)
        return pred
