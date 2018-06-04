import cv2
import matplotlib.pylab as plt
import numpy as np
from numpy.core.tests.test_mem_overlap import xrange


def brute_force_matcher():
    """
        BF 匹配，首先用cv2.BFMatcher()创建改对象。
            normType:使用的测量距离。默认是cv2.NORM_L2,对SIFT和SURF很好。
                    对应二值字符，ORB,BRIEF,BRISK使用cv2.NORM_HAMMING
                    如果ORBVTA_K==3或4，cv2.NORM_HAMMING2将会被使用。
            crossCheck:默认是False。
                        如果是True,匹配器将会返回匹配值(i,j)如集合A中第i个和集合B中第J个描匹配。反之亦然。
                        也就是两个集合特性应该相互匹配。并两个一致的效果，这是比率测试的很好选择。

        一旦创建好对象。两个重要的方法是BFMatcher.match()和BFMatcher.knnMatch()
            第一个方法返回最佳匹配。
            第二个方法返回k的最佳匹配，k是用户来提供的。当我们需要额外添加时，这个很好的。
        就行我们使用cv2.drawKeypoints()。cv2.drawMatched()能够帮我们画出匹配。
            画出两个水平线，第一个图片和第二个图片被相连。
        cv2.drawMatchedKnn能够画出所有k最好匹配。
            如果k=2,将会每个匹配点画出 2条最佳匹配线。所以要传递mask如果我们要选择性画出
        将会SURF和ORB
    :return:
    """
    pass


def brute_force_matching_with_ORB_descriptors():
    """
    将会看到如果匹配两张图片的例子。
    matches=bf.match(des1,des2)一系列DMatch对象。DMatch有很多属性:
    1.DMatch.distance:描述符之间距离，越近越好
    2.DMatch.trainIdx:描述符在训练集中索引
    3.DMatch.queryIdx:查询描述集中索引
    4.DMatch.imgIdx  :训练集中索引
    :return:
    """
    img1 = cv2.imread('image/box.jpg', 0)
    img2 = cv2.imread('image/box_in_all.jpg', 0)
    outImage = img1.copy()

    # initiate ORB detector
    orb = cv2.ORB_create()
    # orb = cv2.ORB()

    # find the keypoints and descriptions with orb
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match descriptors
    matches = bf.match(des1, des2)

    # sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # draw first 10 matches
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], outImage, flags=2)

    plt.imshow(img3), plt.show()


def brute_force_matching_wit_SIFT_descriptors_and_ratio_test():
    """
        BFMatcher.knnMatch()获得k值最适合匹配。这个例子k=2进行比率测试
    :return:
    """
    img1 = cv2.imread('image/box.jpg', 0)
    img2 = cv2.imread('image/box_in_all.jpg', 0)
    outImg = img2.copy()

    # initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMather with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, outImg, flags=2)

    plt.imshow(img3), plt.show()


# brute_force_matching_with_ORB_descriptors()
# brute_force_matching_wit_SIFT_descriptors_and_ratio_test()


def FLANN_based_matcher():
    """
        FLANN基本匹配，需要传递两个参数：
            第一个索引参数，哪个算法被使用。
                index_params = dict(algorithm=cv2.FLANN_INDEX_KDIREE, trees=5)
                    当使用ORB，使用
                    index_params=dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6, #12
                            key_size=12,    #20
                            multi_probe_level=1) #2
            第二个参数搜索参数。
                指定索引中遍历的次数。精度越高，时间越多。
                search_params=dict(check=100)
    :return:
    """

    img1 = cv2.imread('image/box.jpg', 0)
    img2 = cv2.imread('image/box_in_all.jpg', 0)

    # initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptor with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDIREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDIREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(des1, des2, k=2)
    matches = flann.knnMatch(des1, des2, k=2)

    # need to draw only good matches ,so create a mask
    matchMask = [[0, 0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matchMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchMask,
                       flags=0)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3), plt.show()

FLANN_based_matcher()