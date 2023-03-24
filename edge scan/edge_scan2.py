import os
import cv2
import numpy as np
from utils2 import *

'''
可以使用二值和高斯两种方法
2.0先高斯再二值好用
method = 'bin' 使用手动二值
method = 'bin2' 使用自动二值
method = 'gauss' 使用高斯
使用说明:
# 按下“s”键保存图片
# 按下“d”键时切换下一张图像
# 按下“a”键时切换上一张图像
# 按下“q”键时退出程序
修改记录：
修改了按键切换图片的方法，使用了cv2.waitKey(0)函数
'''
# 显示信息函数
def show_msg(msg, stackedImage):
    cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                  (1100, 350), (0, 255, 0), cv2.FILLED)
    cv2.putText(stackedImage, msg,
                (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.imshow('Result', stackedImage)
    cv2.waitKey(300)

# 获取边缘
def edge_scan(pathImage=None, heightImg=640, widthImg=480,method='bin',webCamFeed=False):
    indexImg = 0  # 图片的索引
    img_list = getFileList(pathImage)
    initializeTrackbars()  # 初始化阈值滑动条
    while True:
        # 1、读取图片或开启摄像头
        if webCamFeed:
            cap = cv2.VideoCapture(0)
            cap.set(10, 160)  # 亮度
            success, img = cap.read()
        else:
            img = cv2.imread(os.path.join(pathImage, img_list[indexImg]))
            # 获取原图片长宽
            # ori_imgHeight, ori_imgWidth, _ = img.shape
        # 2、图像预处理
        img = cv2.resize(img, (widthImg, heightImg))  # 调整图片大小
        thres = valTrackbars()  # 实例化阈值的滑动条
        # 2.1、数字图像处理
        imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # 如果需要，为测试调试创建空白图像
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 2)  # 高斯滤波
        # 2.2 二值和高斯两种方法
        if method == 'bin':
            ret, imgbin = cv2.threshold(imgGray, thres[0], 255, cv2.THRESH_BINARY)  # 二值化
        elif method == 'bin2':
            ret, imgbin = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
        elif method == 'gauss':
            imgbin = imgBlur
        imgCanny = cv2.Canny(imgbin, thres[1], thres[2])  # 边缘检测器
        kernel = np.ones((5, 5))  # 图像处理的卷积核
        imgDial = cv2.dilate(imgCanny, kernel, iterations=thres[3])  # 图像处理：膨胀 粗化边界
        imgCanny = cv2.erode(imgDial, kernel, iterations=thres[4])  # 图像处理：腐蚀  膨胀腐蚀可以帮助我们消除缝隙和杂物
        #闭运算
        # imgCanny = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel)

        # 3、找到轮廓
        imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # 绘制所有检测到的轮廓
        # 3.1、 找到轮廓中的最大轮廓
        biggest, maxArea = biggestContour(contours, maxArea=thres[5], epsilon=thres[6])  # #判断是否为最大矩形
        if biggest.size != 0:
            biggest = reorder(biggest)  # 重新排序
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # 绘制最大轮廓
            imgBigContour = drawRectangle(imgBigContour, biggest, 2)  # 绘制最大轮廓的外接矩形
            # 3.2、对最大轮廓进行仿射变换
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # 仿射变换

            # 3.3、相当于切除不必要部分
            imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
            imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
            # 4、用于显示的图像阵列，即你要显示什么操作的结果，例如高斯模糊
            # ps 只有找到最大矩形时以下才会执行
            if method == 'bin'or method == 'bin2':
                imageArray = ([img, imgbin, imgBlur, imgCanny],
                              [imgContours, imgBigContour, imgWarpColored, imgBlank])
                lables = [["ori", "Bin", "imgBlur", "imgCanny"],
                          ["imgContours", "imgBigContour", "out", ""]]
            else:
                imageArray = ([img, imgGray, imgBlur, imgCanny],
                          [imgContours, imgBigContour, imgWarpColored, imgBlank])
                lables = [["ori", "Gray", "imgBlur", "imgCanny"],
                          ["imgContours", "imgBigContour", "imgWarpColored", ""]]
        # 只有找不到时以下才会执行
        else:
            # 4.1、显示结果
            if method == 'bin' or method == 'bin2':
                imageArray = ([img, imgbin, imgBlur, imgCanny],
                              [imgBlank, imgBlank, imgBlank, imgBlank])
                lables = [["ori", "Bin", "imgBlur", "imgCanny"],
                          ["", "", "", ""]]
            else:
                imageArray = ([img, imgGray, imgBlur, imgCanny],
                              [imgBlank, imgBlank, imgBlank, imgBlank])
                lables = [["ori", "Gray", "imgBlur", "imgCanny"],
                          ["", "", "", ""]]


        # 5、标签显示
        stackedImage = stackImages(imageArray, 0.75, lables)  # 显示标签
        cv2.imshow("Result", stackedImage)

        # ！6、按键操作
        # 按下“s”键保存图片
        # 按下“d”键时切换下一张图像
        # 按下“a”键时切换上一张图像
        # 按下“q”键时退出程序
        key = cv2.waitKey(1)&0xFF
        if key  == ord('s'):
            mkdir(os.path.join(pathImage, "Scanned"))  # 创建保存目录
            # 获取当前图片名字
            c_img = os.path.basename(img_list[indexImg])
            cv2.imwrite(os.path.join(pathImage, "Scanned", c_img), imgWarpColored)
            # 显示保存
            msg = "Scan Saved"
            show_msg(msg, stackedImage)
        elif key  == ord('d'):
            indexImg += 1
            if indexImg == len(img_list):
                indexImg = 0
            # 显示下一张
            msg = "Next Image"
            show_msg(msg, stackedImage)
        elif key  == ord('a'):
            indexImg -= 1
            if indexImg == 0:
                indexImg = len(img_list)
            # 显示上一张
            msg = "Last Image"
            show_msg(msg, stackedImage)
        elif key  == ord('q'):
            break


if __name__ == '__main__':
    webCamFeed = False  # True 开启摄像头
    pathImage = r'D:\Files\Learning\cv_test\test_imgs'  # 关闭摄像头时图片文件夹
    heightImg = 320
    widthImg = 320
    method='bin'  # bin 二值化
    edge_scan(pathImage, heightImg, widthImg,method)
