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
    in:
        1.image
        2.retrieval mode:检索方式
        3.contour approximation 轮廓近似法
            轮廓线优化的方法：
                cv2.CHAIN_APPROX_NONE：如直线轮廓线是直线，需要保存所有点，但其实两点就可以满足了。
                cv2.CHAIN_APPROX_SIMPLE：就是为了去除冗余，节约内存
    out:
        1.output image
        2.contours
        3.hierarchy:等级
    :return:
    """
    img = cv2.imread('image/self.jpg')
    image_or = cv2.imread('image/dog.png')
    image_gray = cv2.cvtColor(image_or, cv2.COLOR_BGR2GRAY)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    image, contours, hierarchy, = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 画图中所有的轮廓
    img_all = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
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


def contour_properties():
    """
    获取常用的属性：Solidity体积，Equivalent Diameter等效直径,mask image掩膜像，mean intensity平均强度
    :return:
    """
    img = cv2.imread('image/test.jpg', 0)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    image, contour, hierarchy = cv2.findContours(thresh, 1, 2)

    # 1.aspect ratio(横纵比)
    cnt = contour[0]
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_radio = float(w) / h

    # 2. extent：extent=object area/bounding rectangle area
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    extent = float(area) / rect_area

    # 3. solidity :solidity=counter area/convex hull area
    area1 = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area

    # 4.equivalent diameter：面积和轮廓面积相等原的直径
    # equivalent diameter=sqrt(4*counter_area)/pi
    area2 = cv2.contourArea(cnt)
    equi_diameter = np.sqrt(4 * area2 / np.pi)

    # 5.orientation：物体指向的方向，主轴和次轴的长
    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

    # 6.mask and pixel points：获取组成物体的所有点
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    pixelpoints = np.transpose(np.nonzero(mask))
    # pixelpoints=cv2.findNonZeros(mask)

    # 7.最大值，最小值和他们的坐标
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img, mask=mask)

    # 8.mean color or mean intensity
    mean_val = cv2.mean(img, mask=mask)

    # 9.extreme points极点：topmost,bottommost,rightmost,leftmost
    leftmost = tuple(cnt[cnt[:, :0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommos = tuple(cnt[cnt[:, :, 1].argmax()][0])


def more_functions():
    """
         1.凸性缺陷(convexity defect)和如何找到他们
         2.一点到宁一点的最短距离(多边形)
         3.匹配不同的形状

         1.凸性缺陷：任何和包络线有偏差的缺陷都是凸性缺陷

    :return:
    """
    img_src = cv2.imread('image/meisi.jpg')
    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    image, contour, hierarchy = cv2.findContours(thresh, 2, 1)
    cnt = contour[0]
    # 1.returnPoints=False为了找缺陷，需要False
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    # 2.point polygon test:找出轮廓线到图片中点的距离。轮廓线以内为正值，以外为负值
    dist = cv2.pointPolygonTest(cnt, (50, 50), True)
    # True:找到标记的距离
    # False:返回+1,-1,0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        print(start, end, far)

        cv2.line(img, start, end, [0, 255, 0], 2)
        cv2.circle(img, far, 5, [0, 0, 255], -1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def match_shapes():
    """
       两个图形的匹配度，越小，匹配度越高。
    :return:
    """
    img1 = cv2.imread('image/star.jpg', 0)
    img2 = cv2.imread('image/dog.png', 0)

    ret, thresh = cv2.threshold(img1, 127, 255, 0)
    ret, thresh2 = cv2.threshold(img2, 127, 255, 0)
    image, contours1, hierarchy = cv2.findContours(thresh, 2, 1)
    cnt1 = contours1[0]
    image, contours1, hierarchy = cv2.findContours(thresh2, 2, 1)
    cnt2 = contours1[0]

    ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
    print(ret)


# contours()
# contour_feature()
# contour_properties()
# more_functions()
match_shapes()
