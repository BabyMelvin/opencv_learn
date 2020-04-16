import cv2
import numpy as np
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


def show_image1():
    # 加载一个图片
    img = cv2.imread('image/test.jpg', 1)
    print(img.shape)  # {(400,300,3) -> -1,(400,300)->0,(400,300,3)->1}
    # 路径不会抛出错误，但print img ->None
    # print(img)
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


# show_image1()


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
    if k == 27:  # 等待ESC
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
