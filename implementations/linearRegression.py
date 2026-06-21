import numpy as np

def predict(x, w, b):
    return w * x + b;

def cost(x, y, w, b):
    m = len(x);
    predictions = predict(x, w, b);
    cost = np.sum( ( predictions - y )** 2 )  / (2 * m);
    return cost;

def compute_gradient(x, y, w, b):
    m = len(x);
    predictions = predict(x, w, b);
    dj_dw = np.sum((predictions - y)  * x) / m;
    dj_db = np.sum(predictions  - y ) / m;
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, alpha, num_iters):
    costHistory = [];
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b);
        w = w - alpha * dj_dw;
        b = b - alpha * dj_db;
        costHistory.append(cost(x, y, w, b));
    return w, b, costHistory;


x = np.array([1,2,3,4,5]);
y = np.array([3,5,7,9,11]);
print(x,y);

print(predict(x,2,1));
print(cost(x,y,2,1));

print(cost(x, y, 0, 0))
print(cost(x, y, 2, 1))

print(compute_gradient(x, y, 2, 1))   # perfect fit, gradients should be ~0
print(compute_gradient(x, y, 0, 0))   # bad fit, gradients should be sizable negative numbers


w_final, b_final, history = gradient_descent(x, y, 0, 0, 0.01, 1000)
print(w_final, b_final)
print(history[:5])   # first few costs
print(history[-5:])  # last few costs
