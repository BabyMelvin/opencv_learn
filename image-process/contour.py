import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    contour:等高线(轮廓)
        寻找等到线和画等高线
    等到线，轮廓线：
        沿着边界连续点连起来形成的曲线，轮廓线。
        作用:
            形状分析
            物体检测识别    
        1.等高线之前用阀值过滤或者Canny edge边缘检测，使用二进制图精度高。
        2.寻找等高线会去修改图像内容，使用之前要备份
        3.opencv中是在从黑背景寻找白色物体过程
"""


def contours():
    """
    1.image
    2.retrieval mode:检索方式
    3.contour approximation 轮廓近似法
    out:
        1.output image
        2.contours
        3.hierarchy:等级
    :return:
    """
    img = cv2.imread('self.jpg')
    image_or = cv2.imread('dog.png')
    image_gray = cv2.cvtColor(image_or, cv2.COLOR_BGR2GRAY)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    image, contours, hierarchy, = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 画图中所有的轮廓
    img_all = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # 画某一个，比如第4个
    img_4th = cv2.drawContours(img, contours, 3, (0, 255, 0), 3)
    # 或者
    # cnt = contours[2]
    # image_2th = cv2.drawContours(image, [cnt], 0, (0, 255, 0), 3)
    plt.subplot(221), plt.imshow(image_or), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(image_gray, cmap='gray'), plt.title('gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_all), plt.title('contour_all')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(img_4th), plt.title('contour 4th')
    plt.xticks([]), plt.yticks([])
    plt.show()


def contour_feature():
    """
        轮廓不同特点：
            面积，周长(perimeter)，图心(centroid),边界框
    :return:
    """
    img = cv2.imread('test.jpg', 0)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    image, contour, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contour[0]
    # 1.Moments,图形中心和面积 Cx=int(M['m10']/M['m00']),Cy=int(M['m01']/M['m00'])
    M = cv2.moments(cnt)
    print(M)
    # 2. 面积或者M['m00']
    area = cv2.contourArea(cnt)
    # 3. 周长,shape is closed:True or False
    perimeter = cv2.arcLength(cnt, True)
    # 4. 轮廓近似值
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # 5.凸包络线
    # points:传进去的点
    # hull  :输出，经常忽略
    # clockwise:方向， True顺时针，False逆时针
    # returnPoints:默认是True，
    hull = cv2.convexHull(cnt)
    # 6.检查是否是图面体
    k = cv2.isContourConvex(cnt)
    # 7.边框矩形   1.直边框矩形 2.旋转长方形
    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 2)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int(box)
    img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    # 8.最小包圆
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)
    # 9.适配椭圆
    ellipse = cv2.fitEllipse(cnt)
    im = cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    # 10.适配线
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int(-x * vy / vx + y)
    righty = int(((cols - x) * vy / vx) + y)
    img = cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)


# contours()
contour_feature()
