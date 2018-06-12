import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit

"""
    目标:
        1. 寻找图片中傅里叶变换
        2. 利用numpy中的FFT
        3. 傅里叶变换的一些应用
    原理：
        傅里叶变换是用来分析频域特征
"""


def fourier_transform_in_numpy1():
    """
    使用np.fft.fft2()以一个复数数组来表示频率变换
    第一个参数：输入图片（灰度图）
    第二个参数：可选，决定输出数组大小。
        如果数组大于输入，那么在FFT计算之前用0扩充边界。
        如果大小小于输入图片，输入图片会被裁剪。
        如果没有输入，则输出和输入同样大。

    一旦得到结果，0频（DC）成分将位于左上角。
    如果想要0频道中心点，需要两个方向对结果移动N/2。通过np.fft.fftshift()完成。
    更容易分析。一旦完成频率转换，你就获得振幅谱
    :return:
    """
    img = cv2.imread('image/meisi.jpg', 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


# 结果将会在中心位置看到更多白色区域，代表低频内容。
# fourier_transform_in_numpy1()
def fourier_transform_in_numpy2():
    """
    有了频域变换，就可以做很多频率的操作。比如：高通过滤，重构图片，找到逆变换DFT。
    可以简单利用60x60方型窗口，去除低频。然后利用逆变换np.fft.ifftshift()DC部分又回到左上角。
    然后用np.ifft2()来找到逆变换。结果还是一个复数,能够取绝对值
    :return:
    """
    img = cv2.imread('image/meisi.jpg', 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = int((rows - 1) / 2), int((cols - 1) / 2)
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_back)
    plt.title('result in JET'), plt.xticks([]), plt.yticks([])
    plt.show()


# 结果表明高通滤波是一种边缘检测操作。同样证明图片大部分是低频显示。
# fourier_transform_in_numpy2()

def fourier_transform_in_opencv1():
    """
    在上面JET中可看到一些人工品（一些阴影）。
    显示些波纹状，被称为ringing effets.因为我们使用窗口掩盖的原因。mask和正弦卷积会导致这个情况。
    方型窗口一般不用来过滤，最好使用高斯窗口。

    cv2.dft()和 cv2.idft()函数。和上面相同，返回两通道。
    第一个通道：
        结果实部
    第二通道：
        结果的虚部

    输入图片首先变换成np.float32位
    :return:
    """
    img = cv2.imread('image/meisi.jpg', 0)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


# fourier_transform_in_opencv1()
def fourier_transform_in_opencv2():
    """
    可以使用 cv2.cartToPolar()返回一个点的幅值和相位
    现在要做DFT逆变换。之前我们建立HPF，这次我们移除高频部分。简单使用LPF。
    实际上使图片模糊。这次，我们建立一个mask，低频用1，LF。高频PF用0.
    :return:
    """
    img = cv2.imread('image/meisi.jpg', 0)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = int(rows - 1 / 2), int(cols / 2)
    # create a mask first,center square is 1,remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('Magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


# fourier_transform_in_opencv2()

def performance_optimization_of_DFT():
    """
    DFT计算性能对某些数组尺寸效果很好当数组尺寸使2的倍数可能更快。
    数组是2，3,5倍数也很好效率。如果想用好的效率，添加0边界，来获得更好的效率。
    numpyFFT 输入新的尺寸，默认加0边界。
    :return:
    """
    img = cv2.imread('image/meisi.jpg', 0)
    rows, cols = img.shape
    print(rows, cols)
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    print(nrows, ncols)
    # TODO 增加0边界
    # nimg = np.zeros(nrows, ncols)
    # nimg[:rows, :cols] = img
    # 增加边界，或者
    right = ncols - cols
    bottom = nrows - rows
    bordertype = cv2.BORDER_CONSTANT  # just to avoid line breakup in PDF file
    nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, bordertype, value=0)

    # 计算DFT性能，与Numpy函数作比较。 有问题？？？
    timeit.timeit('fft1=np.fft.fft2(img)', setup='import numpy as np;import cv2;img=cv2.imread("image/meisi.jpg")')
    timeit.timeit('fft2=np.fft.fft2(img,[nrows,ncols])', setup='import numpy as np')


# performance_optimization_of_DFT()
def why_Laplacian_high_pass_filter():
    # simple averaging filter without scaling parameter
    mean_filter = np.ones((3, 3))

    # creating a gussian filter
    x = cv2.getGaussianKernel(5, 10)
    gaussian = x * x.T

    # different edge detecting filters
    # scharr in x-direction
    scharr = np.array([
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]
    ])

    # sobel in x direction
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # sobel in y direction
    sobel_y = np.array([
        [-1, 2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # laplacian
    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
    filter_names = ['mean_filter', 'gaussian', 'laplacian', 'sobel_x', 'sobel_y', 'scharr']
    fft_filters = [np.fft.fft2(x) for x in filters]
    mag_spectrum = [np.log(np.abs(z) + 1) for z in fft_filters]

    for i in np.arange(6):
        plt.subplot(2, 3, i + 1), plt.imshow(mag_spectrum[i], cmap='gray')
        plt.title(filter_names[i]), plt.xticks([]), plt.yticks([])
    plt.show()


why_Laplacian_high_pass_filter()
