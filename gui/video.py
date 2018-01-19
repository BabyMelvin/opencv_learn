import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

"""
    目标:
        1.学习读,显示,保存 视频
        2.学会用照相机抓图，并显示
        3.cv2.VideoCapture(),cv2.VideoWriter()
"""


def capture_video_from_camera():
    """
    通常使用相机抓取实时流。
    用笔记本内置webcam，转换成灰度视频并显示
    :return:
    """
    # 参数：device index(0 or -1,1:第二个) or name of a video file
    # 逐帧捕捉，最后要释放
    cap = cv2.VideoCapture(0)
    # cap.get(prop)
    # cap.set(pro,value)
    while True:
        # 逐帧捕捉
        if cap.isOpened():
            ret, frame = cap.read()
        else:
            sleep(2)
            print("摄像头未能打开")
            continue
        # 对帧操作
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print("摄像头读帧失败")
            continue

        # 显示生成框架
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 结束释放
    cap.release()
    cv2.destroyAllWindows()


def playing_video_from_file():
    """
    传入VideoCapture(文件名)
    可用慢动作播放，25毫秒是正常的速度
    :return:
    """
    count = 0
    cap = cv2.VideoCapture('test.avi')
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)
            # ord Return the Unicode code point for a one-character string.
            print("playing.....")
            # cv2.waitKey(0)
            if (cv2.waitKey(0) & 0xFF) == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def saving_video():
    """
         image:cv2.imwrite()
         video:
            FourCC code:
                FourCC 4位用来识别视频编码
                依赖平台的
                window:DIVX
                cv2.VideoWriter_fourcc('M','J','P','G') or cv2.VideoWriter_fourcc(*'MJPG') for MJPG
            每帧显示时间fps:
            每帧大小frame size:
            isColor:
                true:编码需要彩色帧
                false:灰度帧
    :return:
    """
    cap = cv2.VideoCapture(0)
    # 定义codec和创建VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 0)
            # 写入翻转的帧
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 释放所有
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# capture_video_from_camera()
# playing_video_from_file()
saving_video()
