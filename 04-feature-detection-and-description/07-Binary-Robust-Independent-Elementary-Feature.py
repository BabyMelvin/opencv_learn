import cv2
import matplotlib.pylab as plt
import numpy as np


def brief_in():
    img = cv2.imread('image/cute.jpg', 0)

    # initiate STAR detector
    star = cv2.FeatureDetector_create("STAR")

    # initiate BRIEF extractor
    brief = cv2.DescriptorExtractor_create("BRIEF")

    # find the keypoints with STAR
    kp = star.detect(img, None)

    # compute the descriptor with BRIEF
    kp, des = brief.compute(img, kp)

    print(brief.getInt('byte'))
    print(des.shape)


brief_in()
