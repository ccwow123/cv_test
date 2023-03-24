# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np


def nothing(x):
    pass
def init_Trackbars():
    cv2.createTrackbar("p1", "image", 100, 255, nothing)#Canny边缘检测器的高阈值 默认值100
    cv2.createTrackbar("p2", "image", 100, 255, nothing)#默认值100
    cv2.createTrackbar("minRadius", "image", 0, 255, nothing)
    cv2.createTrackbar("maxRadius", "image", 0, 20, nothing)
    cv2.createTrackbar("minDist", "image", 100, 200, nothing)

def get_val():
    Threshold1 = cv2.getTrackbarPos("p1", "image")
    Threshold2 = cv2.getTrackbarPos("p2", "image")
    Threshold3 = cv2.getTrackbarPos("minRadius", "image")
    Threshold4 = cv2.getTrackbarPos("maxRadius", "image")
    Threshold5 = cv2.getTrackbarPos("minDist", "image")
    return Threshold1, Threshold2, Threshold3, Threshold4, Threshold5

if __name__ == '__main__':
    cv2.namedWindow('image')#创建一个窗口
    init_Trackbars()#初始化滑动条
    img = cv2.imread(r"test_imgs/1.jpg")#读取图片
    # p1 = 71 ,p2 =83  对应2.jpg最好
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换为灰度图
    imgBlur = cv2.GaussianBlur(img_gray, (5, 5), 2)# 高斯滤波
    while True:
        low_threshold, high_threshold, minRadius, maxRadius, minDist = get_val()#获取滑动条的值
        circles = cv2.HoughCircles(imgBlur, cv2.HOUGH_GRADIENT, 1, minDist, param1=low_threshold, param2=high_threshold,
                                   minRadius=minRadius, maxRadius=maxRadius)#霍夫圆检测
        img_show = img.copy()
        if circles is not None:#如果检测到圆
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(img_show, (i[0], i[1]), i[2], (0, 255, 0), 2)#画圆
                cv2.circle(img_show, (i[0], i[1]), 2, (0, 0, 255), 3)#画圆心
                cv2.putText(img_show, "R:{}".format(i[2]), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)#显示半径
        cv2.imshow('image', img_show)
        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('s'):
            cv2.imwrite("result.jpg", img_show)

