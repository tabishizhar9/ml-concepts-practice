import numpy as np

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    return (1 / (2 * m)) * cost_sum


# test it
w = 100
b = 100
cost = compute_cost(x_train, y_train, w, b)

print("Cost:", cost)