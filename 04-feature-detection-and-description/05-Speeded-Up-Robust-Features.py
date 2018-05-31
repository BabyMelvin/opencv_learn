import cv2
import matplotlib.pylab as plt
import numpy


def speed_up_robust_feature():
    """
        OpenCV提供SURF功能像SIFT一样。
        初始化SURF对象：
            可选条件64/128尺寸描述
            Upright/NormalSURF等
    :return:
    """
    img = cv2.imread('image/ball.jpg', 0)

    # create SURF object. You can specify params here or later
    # here I set Hessian Threshold to 400
    surf = cv2.SURF(400)

    # find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(img, None)
    print(len(kp))
    # 1199关键点显示图片太多了。减少到50个。再图片匹配需要更多特征，但不是现在。
    # 所以增加Hessian Threshold

    # check present Hessian threshold
    print(surf.hessianThreshold)

    # we set it to some 50000.remeber,it is just for representing in picture
    # in actual cases ,it is better to have a value 300-500
    surf.hessianThreshold = 50000
    # again compute keypoints and check its number
    kp, des = surf.detectAndCompute(img, None)
    print(len(kp))

    # 现在少于50，在图片上画出来
    img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    plt.imshow(img2), plt.show()
    # 结果像斑点加测试


def upright_surf():
    """
        使用U-SURF，不会检测方向
    :return:
    """
    # check upright flag,if it False,set it to True
    img = cv2.imread('image/ball.jpg', 0)

    # create SURF object. You can specify params here or later
    # here I set Hessian Threshold to 400
    surf = cv2.SURF(400)

    # find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(img, None)
    print(len(kp))
    # 1199关键点显示图片太多了。减少到50个。再图片匹配需要更多特征，但不是现在。
    # 所以增加Hessian Threshold

    # check present Hessian threshold
    print(surf.hessianThreshold)

    # we set it to some 50000. remember,it is just for representing in picture
    # in actual cases ,it is better to have a value 300-500
    surf.hessianThreshold = 50000

    # check upright flag,if it False,set it to True
    print(surf.upright)

    surf.upright = True

    # recompute the feature points and draw it
    kp = surf.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)

    # 最后检测描述检测尺寸，如果只是64，改成128
    # find size of descriptor
    print(surf.descriptorSize())  # 64

    # that means flag,"extended" is False
    print(surf.extended)

    # so we make it the True to get 128-dim descriptor
    surf.extended = True
    kp, des = surf.detectAndCompute(img, None)
    print(surf.descriptorSize)  # 128
    print(des.shape)
    plt.imshow(img2), plt.show()
    # 结果表明所有的方向是相同的，比上面的速度更快。方向不相关，这个效果比较好。


speed_up_robust_feature()
upright_surf()
