import cv2
import matplotlib.pylab as plt
import numpy as np


def kNN_in_OpenCV():
    """
    红色家庭为Class-0,蓝色家庭表示Class-1:
    25个家庭或25个训练数据。标记为0或1.这些有numpy随机数来产生
    :return:
    """
    # Feature set containing (x,y) values of 25 known/training data
    trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

    # labels each one either Red of Blue with number 0 and 1
    responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

    # take Red families and plot them
    red = trainData[responses.ravel() == 0]
    plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

    # take Blue families and plot them
    blue = trainData[responses.ravel() == 1]
    plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

    # 接下来初始化kNN算法。传递训练数据来训练算法。
    # 引进一个新数据，来用kNN划分。我们数据一个浮点数组，尺寸为tesdataxfeatures
    # 1.如果k=1，得到最近的了。
    # 2.kNN的标记
    # 3.各个最近家族的距离

    # new-comer is marked as green color
    newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
    plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')

    knn = cv2.ml.KNearest_create()
    knn.train(trainData, responses)
    ret, results, neighbours, dist = knn.find_nearest(newcomer, 3)

    print("result", results, "\n")
    print("neighbours:", neighbours, "\n")
    print("distance:", dist)
    plt.show()


def OCR_With_hand_write():
    """
    :return:
    """
    img = cv2.imread('image/digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # now we split the image to 5000 cells ,each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # make it into a Numpy array,It size will be (50,100,20,20)
    x = np.array(cells)

    # now we prepare train_data and test_data
    train = x[:, :50].reshape(-1, 400).astype(np.float32)  # size=(2500,400)
    test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # size =(2500,400)

    # create labels for train and test data
    k = np.arange(10)
    trains_labels = np.repeat(k, 250)[:, np.newaxis]
    test_labels = trains_labels.copy()

    # initiate kNN,train the data ,then test it with test data for k=1
    knn = cv2.ml.KNearest_create()
    knn.train(train, trains_labels)
    ret, result, neighours, dist = knn.find_nearest(test, k=5)

    # now we check the accuracy of classification
    # for that ,compare the result with test_labels and check which are wrong
    matches = result == test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    print(accuracy)

    # instead of finding the training,next time reuse need to save

    # save data
    np.savez('knn_data.npz', train=train, trains_labels=trains_labels)

    # noew load the data
    with np.load('knn_data.npz') as data:
        print(data.files)
        train = data['train']
        trains_labels = data['train_labels']


def OCR_of_English():
    """
    :return:
    """
    # load the data,converters the letter to a number
    data = np.loadtxt('letter-recognition.data', dtype='float32', delimiter=',',
                      converters={0: lambda ch: ord(ch) - ord('A')})

    # split the data to two ,1000 each for train and test
    train, test = np.vsplit(data, 2)

    # split trainData and testData to features and responses
    responses, trainData = np.hsplit(train, [1])
    labels, testData = np.hsplit(test, [1])

    # initiate teh kNN,classify,measure accuracy
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, responses)
    ret, result, neighbours, dist = knn.find_nearest(testData, k=5)

    correct = np.count_nonzero(result == labels)
    accuracy = correct * 100.0 / 10000
    print(accuracy)


kNN_in_OpenCV()

# 需要4.4M内存，使用uint8作为精度值为特征，首先将数据转为uint8然后保存它。
# 1.1M这个时候。然后加载，应该转换成float32
OCR_With_hand_write()

OCR_of_English()
