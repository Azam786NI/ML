import numpy as np


def handel_activation(cal, activation):

    if activation == "sigmoid":
        new_cal = []
        for i in cal:
            new_cal.append(i * (1 - i))

        return np.array([new_cal])


def calculate_gradient(input_seq, f_pass, weights, loss, activaton="ReLU"):
    print("=============== bp ===============")
    gradients = []

    weights = weights[::-1]
    f_pass = f_pass[::-1]

    if activaton != "ReLU":
        dl = loss * handel_activation(f_pass[0], activation=activaton)

    else:
        dl = np.array([loss])

    for i in range(len(f_pass)):
        if i != len(f_pass) - 1:
            gradient = dl.T @ np.array([f_pass[i + 1]])
            gradients.append(gradient)

            dl = dl @ np.array(weights[i]).T

            if activaton != "ReLU":
                dl = dl * handel_activation(f_pass[i + 1], activation=activaton)

        else:
            gradient = dl.T @ np.array([input_seq])
            gradients.append(gradient.T)

    return gradients
