import cv2
import matplotlib.pylab as plt
import numpy as np

"""
input:
    1.samples:应该是np.float32，每个特征一个栏
    2.nclusters(K):最终需要聚合类的数据。
    3.criteria:
        终止迭代的条件。这个条件满足，迭代停止。应该为tuple 3（type,max_iter,epsilon）
        cv2.TREM_CRITERIA_EPS:达到精度，停止迭代。epsilon 达到了。
        cv2.TERM_CRITERIA_MAX_ITER:预定迭代次数达到停止
        cv2.TREM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER:条件都满足
    4.attempts:使用不同标签执行初始化的次数。该算法返回最佳紧凑标签。
        compactness作为返回值
    5.flags:flag表示初始化中心如何取的。
        cv2.KMEANS_PP_CENTERS
        cv2.KMEANS_RANDOM_CENTERS
output:
    1.compactness:这个点到四周点的长度集合
    2.labels：是标签数组，0,1
    3.centers:聚合中心数组
"""


def data_with_only_one_feature():
    """
    只有一个特征数据，如：一维。如T-shirt问题用身高来决定尺寸
    :return:
    """
    x = np.random.randint(25, 100, 25)
    y = np.random.randint(175, 255, 25)
    z = np.hstack((x, y))
    z = z.reshape((50, 1))
    z = np.float32(z)

    # plt.hist(z, 256, [0, 256]), plt.show()

    # 现在KMEAN函数。需要约束条件，10次迭代
    # define criteria=(type,max_iter=10,epsilon=1.0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # set flags (just to avoid line break in the code)
    flags = cv2.KMEANS_PP_CENTERS

    # apply KMeans
    compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, flags=flags)

    A = z[labels == 0]
    B = z[labels == 1]

    # 现在画出A红色B蓝色。他们中心点为黄色
    plt.hist(A, 256, [0, 256], color='r')
    plt.hist(B, 256, [0, 256], color='b')
    plt.hist(centers, 32, [0, 256], color='y')
    plt.show()


def data_with_multiple_features():
    """
    采用高和胖作为特征。
    记住前面，将数据变为单独的列向量。特征是列向量，行向量对应于输入的样本。
    例如：测试数据是50x2，50个人高和胖数据。第一列是高，第二列是体重。
    :return:
    """
    X = np.random.randint(25, 50, (25, 2))
    Y = np.random.randint(60, 85, (25, 2))
    Z = np.vstack((X, Y))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # now separate the data ,note the flatten()
    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]

    # plot the data
    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(B[:, 0], B[:, 1], c='r')
    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    plt.xlabel('Height'), plt.ylabel('Weight')
    plt.show()


def color_quantization():
    """
    有3个特征，RGB，我们需要重建成Mx3的举证。
    :return:
    """
    img = cv2.imread('image/home.jpg')
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria number of clusters(k) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
    K = 8
    ret, labels, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # now convert back into uint8,and make original image
    center = np.uint8(center)
    res = center[labels.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# data_with_only_one_feature()
# data_with_multiple_features()
color_quantization()
