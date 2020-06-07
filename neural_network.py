import numpy as np


class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self._weights = 2 * np.random.random([3, 1]) - 1

    def train(self, inputs, outputs, iterations=10000):
        for i in np.arange(iterations):
            output = self.think(inputs)
            error = outputs - output
            adjust = np.dot(inputs.T, error * NeuralNetwork._sigmoid(output))
            self._weights += adjust

    def think(self, inputs):
        inputs = inputs.astype(np.float)
        return NeuralNetwork._sigmoid(np.dot(inputs, self._weights))

    def get_weights(self):
        return self._weights

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_derivative(x):
        return x * (1 - x)


if __name__ == "__main__":
    brain = NeuralNetwork()
    print("Weights - before training:\n", brain.get_weights())
    training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T
    brain.train(training_inputs, training_outputs)
    print("Weights - after training:\n", brain.get_weights())
    print("Output - after training:\n", brain.think(np.array([1, 0, 0])))
