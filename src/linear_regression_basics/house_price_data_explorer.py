import numpy as np
import matplotlib.pyplot as plt

# training set
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([300.0, 500.0, 700.0, 900.0, 1100.0])

# prints the both training sets
print(x_train)
print(y_train)

# prints m (number of training examples)
print(f"number of training examples [ m ] = {len(x_train)}")

# Access the first training example
training_example = 0
print(f"x^{training_example} ,y^{training_example} = ({x_train[training_example]} , {y_train[training_example]})")

# plot the graph
plt.scatter(x_train, y_train, marker="x")

plt.title("house pricing")
plt.xlabel('Size (1000 sqft)')
plt.ylabel('price in $1000')

plt.show()