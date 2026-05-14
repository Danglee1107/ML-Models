import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

np.random.seed(45)
n = 200

class_0 = np.random.randn(100, 2) * 0.5 + np.array([-1.0, -1.0])
class_1 = np.random.randn(100, 2) * 0.5 + np.array([1.0, 1.0])

# Combine
X = np.vstack((class_0, class_1))
y = np.array([-1] * (n // 2) + [1] * (n // 2))

def display_with_fx(data: npt.NDArray[np.float64], w: npt.NDArray[np.float64]) -> None:
    """
    Plots the data points and the decision boundary (fx as a line) on the same plot.
    """
    plt.figure(figsize=(6, 6))
    # Plot class 0 as red circles ('ro'), class 1 as blue triangles ('b^')
    class_0_idx = (y == -1)
    class_1_idx = (y == 1)
    plt.plot(data[class_0_idx, 0], data[class_0_idx, 1], 'ro', alpha=0.8, label='Class 0')
    plt.plot(data[class_1_idx, 0], data[class_1_idx, 1], 'b^', alpha=0.8, label='Class 1')

    # Decision boundary: w0*x + w1*y + w2*bias = 0
    # For bias=0, line: w0*x + w1*y = 0 => y = -(w0/w1)x
    if w.shape == (1, 3):
        w = w.flatten()
    if w[1] != 0:
        x_vals = np.linspace(-3, 3, 100)
        y_vals = -(w[0] * x_vals + w[2]) / w[1] 
        plt.plot(x_vals, y_vals, 'g-', label='fx (decision boundary)')
    else:
        # Vertical line
        x_val = np.full(100, -w[2]/w[0] if w[0] != 0 else 0)
        y_vals = np.linspace(-3, 3, 100)
        plt.plot(x_val, y_vals, 'g-', label='fx (decision boundary)')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.grid(True)
    plt.title('Data and fx (decision boundary)')
    plt.show()

def sign(w, x):
    return np.sign(np.dot(x, w.T))

def is_converged(y_hat: npt.NDArray[np.float64], 
                 y: npt.NDArray[np.float64]):
    return np.array_equal(y_hat.flatten().astype(int), y.flatten().astype(int))

def main() -> None:
    w = np.random.randn(1, 3)
    bias = np.ones((n, 1))
    _X = np.append(X, bias, axis=1)
    lr = 0.01

    display_with_fx(X, w)

    while True:
        mix_idx = np.random.permutation(n)
        for i in range(n):
            xi = _X[mix_idx[i], :]
            yi = y[mix_idx[i]]
            if sign(w, xi) != yi:
                w += lr * (yi * xi)

        if is_converged(sign(w, _X).T, y):
            break
    
    display_with_fx(X, w)

if __name__ == '__main__':
    main()