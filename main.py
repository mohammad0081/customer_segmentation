import numpy as np
import pandas as pd

def init_centroids(X, k):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:k]]
    return centroids


def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K) :
        points = X[idx == k]

        if len(points) > 0:
            centroids[k] = np.mean(points, axis= 0)

    return centroids


def find_closest_centroid(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distances = []
        for j in range(centroids.shape[0]):
            norm_i_j = np.linalg.norm(X[i] - centroids[j])
            distances.append(norm_i_j)

        idx[i] = np.argmin(distances)

    return idx


def run_k_means(X, initial_centroids, max_iters=100):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)

    for i in range(max_iters):
        print('Epoch : ', i)
        idx = find_closest_centroid(X, centroids)
        centroids = compute_centroids(X, idx, K)

    return centroids, idx

def read_data():

    df = pd.read_csv('Data/marketing_campaign_dataset.csv')

    df.drop('Dt_Customer',axis = 1,  inplace= True)

    df['Marital_Status'] =df['Marital_Status'].replace(
        ['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'],
        [0, 1, 2, 3, 4, 5, 6, 7]
    )

    df['Education'] = df['Education'].replace(
        ['Basic', 'Graduation', 'Master', 'PhD', '2n Cycle'],
        [0, 1, 2, 3, 4]
    )

    return np.array(df)


data = read_data()
initial_centroids = init_centroids(data, k=5)

centroids, idx = run_k_means(X= data, initial_centroids= initial_centroids, max_iters=50)
