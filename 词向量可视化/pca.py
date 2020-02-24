import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def eig(matrix):
    '''
    求特征值和特征向量
    Ax = λx 即 |A-λE|x = 0
    (λ缩放矩阵（向量），A也起到缩放向量的作用)
    A matrix
    λ 特征值
    E 单位向量
    x 特征向量
    :param matrix:
    :return:
    '''
    λ, x = 0, 0
    return λ, x


def my_pca(X, k):  # k is the components you want
    # mean of each feature
    n_samples, n_features = X.shape
    mean = X.mean(axis=0)
    # normalization
    norm_X = X - mean
    # 计算各组向量间的协方差
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    return data


if __name__ == '__main__':
    color = np.linspace(0, 1, 6, dtype=np.float)
    sizes = np.ones((6), dtype=np.float)*100
    X = np.random.random((6, 2))
    Y = my_pca(X, 2)
    print(Y)
    plt.scatter(X[:, 0], X[:, 1], c=color, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=color, s=sizes, alpha=0.5, marker='x')
    plt.show()

    # pca = PCA(n_components=1)
    # pca.fit(X)
    # Y1 = -pca.transform(X)
    # print(Y1)
    # plt.scatter(Y1[:, 0], np.linspace(0, 1, 6), c=color, s=sizes, alpha=0.2)
    # plt.show()
