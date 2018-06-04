import cv2
import matplotlib.pylab as plt
import numpy as np


def feature_matching_homography():
    """
        可以从calib3d模块中cv2.findHomography().
            如果传入两个图像的合集，将会找到各自全景转换。
            然后使用cv2.perspectiveTransform()找到物体。

        找到转换信息至少需要四个正确的点。

        当匹配中可能会产生影响结果的一些错误信息。
        解决这个问题，算法RANSAC或者LEAST_MEDIAN。所以好的匹配提供正确的估计被称为内点，
        剩下称为外点。cv2.findHomography()返回一个mask，这个mask具体化内点和外点。
    :return:
    """

    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('image/box.jpg', 0)
    img2 = cv2.imread('image/box_in_all.jpg', 0)

    # initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDIREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDIREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])

        # at least 10 matches来找到物体。否则简单提示信息不够
        # 如果足够匹配，将会抽取匹配点位置在两个图片中。被传递寻找全景信息。
        # 一旦找到一个3x3转换矩阵，将会使用它将转换查询图片和训练图片上进行对应。最后，画出他们
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("not enough matches are found-%d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None

    # 最后，我们画出内点(成功寻找到物体)或者匹配点(如果失败)
    draw_params = dict(matchColor=(0, 255),  # draw matched in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliters
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()


feature_matching_homography()
