import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Literal

np.random.seed(42)
n = 210

label1  = np.array([0 for _ in range(n//3)])
label2  = np.array([1 for _ in range(n//3)])
label3  = np.array([2 for _ in range(n//3)])
labels = np.hstack((label1, label2, label3))

X1 = np.random.randn(n//3, 2) * 0.3 + np.array([0.5, 0.5])
X2 = np.random.randn(n//3, 2) * 0.3 + np.array([1.25, 1.5])
X3 = np.random.randn(n//3, 2) * 0.3 + np.array([1.75, 0.5])
X = np.vstack((X1, X2, X3))
new_point = np.array([1.045, 0.86])

def display():
    plt.plot(X1[:, 0], X1[: , 1], 'ro', alpha= .8)
    plt.plot(X2[:, 0], X2[: , 1], 'b^', alpha =.8)
    plt.plot(X3[:, 0], X3[: , 1], 'gs', alpha =.8)
    plt.plot(new_point[0], new_point[1], 'kx', markersize= 10)
    plt.xlabel("data")
    plt.ylabel("Y")
    plt.xlim(0,2.5)
    plt.ylim(0,2.5)
    plt.grid(True)
    plt.show()

def manhattan_norm(x1: npt.NDArray[np.float64],
                   x2: npt.NDArray[np.float64], axis: int):
    return np.linalg.norm(x1 - x2, axis= axis, ord= 1)

def euclidean_norm(x1: npt.NDArray[np.float64],
                   x2: npt.NDArray[np.float64], axis: int):
    return np.linalg.norm(x1 - x2, axis= axis, ord= 2)

def minkowski(x1: npt.NDArray[np.float64], 
              x2: npt.NDArray[np.float64], axis: int, ord: int):
    return np.linalg.norm(x1 - x2, axis= axis, ord= ord)

def knn(n_neighbors: int, weight: Literal["uniform", "distance"],
        algorithm: Literal["kd-tree", "ball-tree", "brute"] = "brute",
        metric: Literal["manhattan", "euclidean", "minkowski"] = "minkowski",
        p: int  = 2):

    if metric == "euclidean" or (metric == "minkowski" and p == 2):
        dists = euclidean_norm(X, new_point, axis= 1)

    elif metric == "manhattan" or (metric == "minkowski" and p ==1):
        dists = manhattan_norm(X, new_point, axis= 1)
    
    elif metric == "minkowski":
        dists = minkowski(X, new_point, axis= 1, ord= p)
    

    distances = np.column_stack((dists, labels))
    distances = distances[distances[:, 0].argsort()]

    nearest = distances[:n_neighbors, :]
    nearest_labels = nearest[:, -1]

    if weight == "uniform":
        counts = np.bincount(nearest_labels.astype(int))
        target = np.argmax(counts)

    elif weight == "distance":
        if any(nearest[:, 0] == 0): # the newpoint === datapoint
            return int(nearest[nearest[:, 0] == 0][0,1])

        w = 1 / nearest[:, 0]
        class_weighted = np.bincount(nearest_labels.astype(int), weights= w)
        target = np.argmax(class_weighted)

    return target

def main() -> None:
    K = 11
    w = "uniform"
    target = knn(K, w)

    print(target)

    # display()

if __name__ == '__main__':
    main()