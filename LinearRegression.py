import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 50

X = np.random.rand(n, 1)
one = np.ones((n, 1))
x_hat = np.concatenate((one, X), axis= 1)

expected_weights = np.array([2, 3]) # [bias, w1, w2, ...]
y = expected_weights[0] + expected_weights[1] * X + np.random.randn(n, 1) * 0.5

x_smooth = np.linspace(X[:, 0].min(), X[: , 0].max(), 100)
def display(fx):
    plt.plot(X, y, 'ro')
    plt.plot(x_smooth, fx, color = "blue", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.xlim(0, 1)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.show()

def cost(w):
    e = np.sum((y - x_hat @ w) ** 2)
    e /= n
    return e

def main() -> None:
    w = np.zeros((2, 1))
    epochs = 1000
    lr = 0.01
    count = 0
    epsilon = 1e-10
    while count < epochs:
        dw = (x_hat.T @ x_hat) @ w - x_hat.T @ y 
        prev_w = w
        w -= lr * dw

        if cost(prev_w) < cost(w):
            lr /= 10
            w = prev_w
            continue
        
        if np.linalg.norm(dw) < epsilon:
            break
            
        count += 1

    print(f"w: {w}")
    print(f"target: {expected_weights}")
    print(f"cost: {cost(w)}")
    print(f"learning rate: {lr}")
    print(f"epochs: {count}")
    target = w[0] + w[1] * x_smooth
    display(target)

if __name__ == '__main__':
    main()