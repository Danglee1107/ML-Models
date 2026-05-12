import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

# Reproducible random data
np.random.seed(42)
n = 100

# Class 0 centered at (-2, -2)
class_0 = np.random.randn(n // 2, 2) + np.array([-2, -2])

# Class 1 centered at (2, 2)
class_1 = np.random.randn(n // 2, 2) + np.array([2, 2])

# Combine data
X = np.vstack((class_0, class_1))

# Labels
y = np.array([0] * 50 + [1] * 50)

def display_with_fx(data: npt.NDArray[np.float64], w: npt.NDArray[np.float64]) -> None:
    """
    Plots the data points and the decision boundary (fx as a line) on the same plot.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(data[:, 0], data[:, 1], 'ro', alpha=0.8, label='Data')

    # Decision boundary: w0*x + w1*y + w2*bias = 0
    # For bias=0, line: w0*x + w1*y = 0 => y = -(w0/w1)x
    if w.shape == (1, 3):
        w = w.flatten()
    if w[1] != 0:
        x_vals = np.linspace(-3, 3, 100)
        y_vals = -(w[0] / w[1]) * x_vals  # bias is zero
        plt.plot(x_vals, y_vals, 'b-', label='fx (decision boundary)')
    else:
        # Vertical line
        x_val = np.full(100, -w[2]/w[0] if w[0] != 0 else 0)
        y_vals = np.linspace(-3, 3, 100)
        plt.plot(x_val, y_vals, 'b-', label='fx (decision boundary)')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.grid(True)
    plt.legend()
    plt.title('Data and fx (decision boundary)')
    plt.show()


def main() -> None:
    w = np.random.randn(1, 3)
    # print(w)
    bias = np.zeros((n, 1))
    _X = np.append(X, bias, axis=1)
    fx = _X @ w.T
    # print(fx)
    display_with_fx(X, w)

if __name__ == '__main__':
    main()