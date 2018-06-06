import cv2
import matplotlib.pylab as plt
import numpy as np

SZ = 20
bin_n = 16  # number of bins
svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR,
                  svm_type=cv2.ml.SVM_C_SVC,
                  C=2.67,
                  gamma=5.383)
affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


def OCR_hand_written_digits():
    """
    在kNN中，我们直接用使用像素密度特征向量、这次我们使用HOG（Histogram of Oriented Gradient）作为特征向量。
    在这找到HOG之前，我们用二阶矩阵来改变(deskew)像素。
    deskew()定义一个数字图片和抗色偏。
    :return:
    """
    img = cv2.imread('image/digits.png', 0)
    cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

    # first half is trainData,remaining is testData
    train_cells = [i[:50] for i in cells]
    test_cells = [i[50:] for i in cells]

    ############### Now Training  ######################
    deskewed = [map(deskew, row) for row in train_cells]
    hogdata = [map(hog, row) for row in deskewed]
    trainData = np.float32(hogdata).reshape(-1, 64)
    response = np.float32(np.repeat(np.arange(10), 250)[:, np.newaxis])

    svm = cv2.ml.SVM_create()
    svm.train(trainData, response, params=svm_params)
    svm.save('svm_data.dat')

    ################  Now testing  #########################
    deskewed = [map(deskew, row) for row in test_cells]
    hogdata = [map(hog, row) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1, bin_n * 4)
    result = svm.predict_all(testData)

    #################  check Accuray    #######################
    mask = result == response
    correct = np.count_nonzero(mask)
    print(correct * 100.0 / result.size)


def deskew(img):
    """
    before finding the HOG,we deskew the image using its second order moments
    :param img:
    :return:
    """
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()

    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.wrapAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


def hog(img):
    """
    找到每个单元的HOG描述。首先找到Sobel每个单元X和Y方向的导数。
    然后找到每个像素的幅值和方向，这个梯度16位整数来衡量。分成四个子方块。
    每个子方块计算直方图方向(16bins),幅值作为比重。所以子系统将会有向量包含16个值。
    所有的特征向量将会包含64个值。这特征向量用来训练我们数据。
    :param img:
    :return:
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n * ang / (2 * np.pi))

    # divide to 4 sub-squares
    bin_cells = bins[:10, :10], bins[10:, :10], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[10:, 10:]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist
