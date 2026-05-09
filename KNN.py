import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

from mathematics.distance_method import *
from algorithms.kd_tree import KDTree
from algorithms.ball_tree import BallTree
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

def display(query_point):
    plt.plot(X1[:, 0], X1[: , 1], 'ro', alpha= .8)
    plt.plot(X2[:, 0], X2[: , 1], 'b^', alpha =.8)
    plt.plot(X3[:, 0], X3[: , 1], 'gs', alpha =.8)
    plt.plot(query_point[0], query_point[1], 'kx', markersize= 10)
    plt.xlabel("data")
    plt.ylabel("Y")
    plt.xlim(0,2.5)
    plt.ylim(0,2.5)
    plt.grid(True)
    plt.show()

def brute_force(points: npt.NDArray[np.float64],
                query_point: npt.NDArray[np.float64], 
                n_neighbors: int,
                metric = "minkowski", 
                p: int = 2):

    if metric == "euclidean" or (metric == "minkowski" and p == 2):
        dists = euclidean_norm(points, query_point, axis= 1)

    elif metric == "manhattan" or (metric == "minkowski" and p ==1):
        dists = manhattan_norm(points, query_point, axis= 1)
    
    elif metric == "minkowski":
        dists = minkowski(points, query_point, axis= 1, ord= p)

    distances = np.column_stack((dists, labels))
    distances = distances[distances[:, 0].argsort()]

    nearest = distances[:n_neighbors, :]
    nearest_labels = nearest[:, -1]

    return nearest, nearest_labels

def kd_tree(points: npt.NDArray[np.float64],
            query_point: npt.NDArray[np.float64], 
            n_neighbors: int = 1,
            metric = "minkowski",
            p: int = 2):

    kd = KDTree(points)
    if metric == "minkowski":
        node, dist = kd.search(query_point, ord= p)

    elif metric == "manhattan":
        node, dist = kd.search(query_point, ord= 1)

    # Euclidean by default
    else:
        node, dist = kd.search(query_point)

    if node:
        point = node.point
    nearest_label = labels[np.where(np.all(points == point, axis=1))[0][0]]

    return np.array(dist), np.array([nearest_label])

def ball_tree(points: npt.NDArray[np.float64],
              query_point: npt.NDArray[np.float64], 
              n_neighbors: int = 1,
              metric = "minkowski",
              p: int = 2):

    if metric == "minkowski":
        bt = BallTree(points, ord = p)
        point, dist = bt.search(query_point, bt.head,  ord= p) #type: ignore

    elif metric == "manhattan":
        bt = BallTree(points, ord = 1)
        point, dist = bt.search(query_point, bt.head,  ord= 1) #type: ignore

    # Euclidean by default
    else:
        bt = BallTree(points)
        point, dist = bt.search(query_point, bt.head) #type: ignore
        
    nearest_label = labels[np.where(np.all(points == point, axis=1))[0][0]]

    return np.array(dist), np.array([nearest_label])

def knn(data_points: npt.NDArray[np.float64],
        query_point: npt.NDArray[np.float64],
        n_neighbors: int = 1, 
        weight: Literal["uniform", "distance"] = "uniform",
        algorithm: Literal["kd-tree", "ball-tree", "brute"] = "brute",
        metric: Literal["manhattan", "euclidean", "minkowski"] = "minkowski",
        p: int  = 2):

    if algorithm == "kd-tree":
        nearest, nearest_labels = kd_tree(data_points, query_point,n_neighbors,  metric, p)

    elif algorithm == "ball-tree":
        nearest, nearest_labels = ball_tree(data_points, query_point,n_neighbors,  metric, p)

    # Brute Force by default
    else:
        nearest, nearest_labels = brute_force(data_points,query_point, n_neighbors, metric, p)
        

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
    K = 1
    w = "uniform"
    a = "brute"
    q_point = np.array([1.5 , 1.5])
    target = knn(data_points= X, 
                 query_point= q_point,
                 algorithm= a,
                 n_neighbors= K,
                 metric= "euclidean",
                 weight= w)

    print(target)

    # display(q_point)

if __name__ == '__main__':
    main()