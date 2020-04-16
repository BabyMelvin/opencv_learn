import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import time

"""
    目标:
        1.学习读,显示,保存 视频
        2.学会用照相机抓图，并显示
        3.cv2.VideoCapture(),cv2.VideoWriter()
"""


def time_module():
    """
    time 和 calendar 模块
    时间戳来自1970.1.1午夜,只支持到2038年???
    一周几日：0-6
    一年：    1-366（儒略历）
    struct_time 元组有这些属性:
        tm_year
        tm_mon
        tm_mday
        tm_hour
        tm_min
        tm_sec
        tm_wday: 0-6
        tm_yday: 1-366
        tm_isdst
    :return:
    """
    ticks = time.time()
    print("当前的时间戳为:", ticks)
    localtime = time.localtime(time.time())
    print("本地时间", localtime)  # 返回struct_time
    localtime_f1 = time.asctime(time.localtime(time.time()))
    print("本地时间为", localtime_f1)  # Fri Jun  8 21:48:50 2018

    # 格式化时间
    localtime_f2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(localtime_f2)

    time.clock()  # 返回CPU秒数


# time_module()


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
    count = 1

    print("当前时间：", time.asctime(time.localtime(time.time())))
    while True:
        # 前一帧时间
        pre_clock = time.clock()
        # 逐帧捕捉
        if cap.isOpened():
            ret, frame = cap.read()
        else:
            sleep(2)
            print("摄像头未能打开")
            continue
        # 对帧操作
        next_clock = time.clock()
        print("get frame after ", next_clock - pre_clock)
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print("摄像头读帧失败")
            continue

        # 显示生成框架
        # cv2.imshow('frame', gray)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 结束释放
    cap.release()
    cv2.destroyAllWindows()


capture_video_from_camera()


def playing_video_from_file():
    """
    传入VideoCapture(文件名)
    可用慢动作播放，25毫秒是正常的速度
    :return:
    """
    count = 0
    cap = cv2.VideoCapture('image/test.avi')
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)
            print("playing.....")
            time.sleep(0.1)
            # cv2.waitKey(0)

            # 将会阻塞?
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# playing_video_from_file()


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
    out = cv2.VideoWriter('image/output.avi', fourcc, 20.0, (640, 480))
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


# saving_video()
