import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


if __name__ == "__main__":
    TRAIN_ITERATIONS = 20000
    training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T
    np.random.seed(1)
    weights = 2 * np.random.random([3, 1]) - 1
    outputs = np.zeros(4)

    print("Start weights:\n", weights)

    for i in range(TRAIN_ITERATIONS):
        input_layer = training_inputs
        outputs = sigmoid(np.dot(input_layer, weights))
        error = training_outputs - outputs
        adjust = error * sigmoid_derivative(outputs)
        weights += np.dot(input_layer.T, adjust)

    print("End weights:\n", weights)
    print("Output after training:\n", outputs)
