import cv2
from matplotlib import pyplot as plt

"""
    image
        cv2.imread()
        注意cv2是BGR模式，而Matplotlib RGB
            flag:
                -1->cv2.IMREAD_COLOR:加载彩色图片。图像透明度被忽略
                 0->cv2.IMREAD_GRAYSCALE:灰度模式加载
                 1->cv2.IMREAD_UNCHANGED:包含alpha通道
        cv2.imshow()
            显示图片，窗口自适应图片大小。
        cv2.imwrite()
        
        1.浮点算法：Gray=R*0.3+G*0.59+B*0.11
        2.整数方法：Gray=(R*30+G*59+B*11)/100
        3.移位方法：Gray =(R*76+G*151+B*28)>>8
        4.平均值法：Gray=（R+G+B）/3
        5.仅取绿色：Gray=G
"""

"""
图像属性：
OpenCV通过Numpy数组表示图像数据，每个图像都是多维数组。每个元素对应一个像素。
图像的尺寸和颜色模式也可以通过数组的形状来表示。

图像的基本属性：
- 图像的尺寸（Width, Height）：可以通过 img.shape 获取。
- 颜色通道（Channels）：通常为 RGB（三个通道），也可以是灰度图（单通道）。
- 数据类型（Data type）：常见的有 uint8（0-255 范围），也可以是 float32 或其他。
"""

def show_image1():
    # 加载一个图片
    img_1 = cv2.imread('image/test.jpg', -1)
    img_0 = cv2.imread('image/test.jpg', 0)
    img = cv2.imread('image/test.jpg', 1)
    print(img_1.shape) # (400, 300, 3)
    print(img_0.shape) # (400, 300)
    print(img.shape)   # (400, 300, 3)
    print(img_1.shape[0], img_1.shape[::-1])

    print(type(img_1.shape))

    if img is None:
        print("无法加载图像")
        exit()

    # 路径不对抛出错误，但print img ->None

    print(img[:3, :3, :])
    img_part100 = img[:100, :100, :]
    img_part_gray_100 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:100, :100]
    # 显示图片,image100和image_gray100 大小相同，因为像素为都为100x100
    cv2.imshow('image100', img_part100)
    cv2.imshow('image_gray100', img_part_gray_100)

    # 键盘信息,等待键盘按键响应
    # esc:27   space:32
    print(cv2.waitKey(0))
    cv2.destroyAllWindows()


show_image1()


def show_image2():
    img = cv2.imread('image/test.jpg')
    # 也可以先建立一个窗口，然后再加载图片
    # cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('image2',cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow('image2', cv2.WINDOW_FULLSCREEN)
    cv2.imshow('image2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# show_image2()


def save_image():
    img = cv2.imread('image/test.jpg', 0)
    # 保存图片
    cv2.imwrite('image/test_save.png', img)


# save_image()


def show_image3():
    # 完整程序
    img = cv2.imread('image/test.jpg', 0)
    cv2.imshow('image3', img)
    k = cv2.waitKey(0) & 0xFF  # 64位系统0
    # print('k=', k, "ord('s')", ord('s'))
    if k == 27 or k == ord('q'):  # 等待ESC
        cv2.destroyAllWindows()
    elif k == ord('s'):  # 's' key保存和退出 ,键盘
        cv2.imwrite('testgray2.png', img)
        cv2.destroyAllWindows()


# show_image3()


def matplotlib_show_image():
    """
        matplotlib
            显示图片
            zoom images: 缩放图片，变焦
    """
    img = cv2.imread('image/test.jpg')
    plt.title("MyPicture")
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    #plt.xticks([]), plt.yticks([])  # hide tick values on x and y axis
    #plt.legend(loc='right', shadow=True) # label not found
    plt.show()


# matplotlib_show_image()
