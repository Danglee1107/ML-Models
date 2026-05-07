import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# number of data points
n = 100

centroids = np.array([
    [4,2],
    [9,3],
    [5,10]
])

cluster1 = np.random.randn(n, 2) + centroids[0]
cluster2 = np.random.randn(n, 2) + centroids[1]
cluster3 = np.random.randn(n, 2) + centroids[2]


X = np.vstack((cluster1, cluster2, cluster3))

def display(ctr):
    plt.plot(X[:, 0], X[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(ctr[:, 0], ctr[:, 1], 'rs', markersize = 10, alpha = .8)
    plt.title("Datasets")
    plt.xlim(1, 10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def display_elbow(wcss):
    plt.plot(wcss[:, 0], wcss[:, 1])
    plt.title("Elbow method")
    plt.xlim(1, 10)
    plt.xlabel("k")
    plt.ylabel("wcss error")
    plt.show()

def k_meansplus(K):
    num_ctr = 1
    index = np.random.choice(len(X), size= 1)
    ctrs = list(X[index])
    while num_ctr < K:
        distances = []

        for pt in X:
            d = np.min([np.linalg.norm(pt - ctr) ** 2 for ctr in ctrs])
            distances.append(d)

        probs = np.array(distances) / np.sum(distances)
        p = np.random.choice(len(X), p= probs)
        if not any(np.array_equal(X[p], ctr) for ctr in ctrs):
            ctrs.append(X[p])
        else:
            continue
        num_ctr += 1

    return np.array(ctrs)

def k_means(K, epochs= 50, return_wcss= False):
    ctrs = k_meansplus(K)
    ctr_labels = np.array([i for i in range(K)])

    count = 0
    error = []
    while count < epochs:
        labels = []

        for pt in X:
            distances = [np.linalg.norm(pt - ctrs[label]) for label in ctr_labels]
            labels.append(np.argmin(distances)) # append the label of minimum ctr

        labels = np.array(labels)
        new_ctrs = []
        for k in ctr_labels:
            points = X[labels == k]

            if len(points) == 0:
                new_ctrs.append(ctrs[k])
            else:
                new_ctrs.append(np.mean(points, axis= 0))

        new_ctrs = np.array(new_ctrs)

        if np.allclose(ctrs, new_ctrs):
            break

        ctrs = new_ctrs
        count += 1
        
    distances = []
    for k in ctr_labels:
        points = X[labels == k]
        d = [np.linalg.norm(p - ctrs[k]) ** 2 for p in points]
        distances.extend(d)

    error.extend([K,np.sum(distances)])
    if return_wcss:
        return error

    return ctrs

def elbow_method(end=10):
    wcss_list = []
    for k in range(1 , end + 1):
        wcss = k_means(k, return_wcss= True)
        wcss_list.append(wcss)
    wcss_list = np.array(wcss_list)

    display_elbow(wcss_list)

def main() -> None:
    # elbow_method()
    K = 3
    # init_ctrs = k_meansplus(K)
    # display(init_ctrs)

    ctrs = k_means(K, epochs=50)
    display(ctrs)

if __name__ == '__main__':
    main()