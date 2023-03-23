
import cv2
import numpy as np
import utlis

########################################################################
webCamFeed = False
pathImage = r"1.jpg"
cap = cv2.VideoCapture(0)
cap.set(10, 160)  # 亮度
heightImg = 640
widthImg = 480
utlis.mkdir(r"./Scanned")#创建保存目录
########################################################################

utlis.initializeTrackbars()
count = 0

while True:

    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE

    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # 如果需要，为测试调试创建空白图像
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
    thres = utlis.valTrackbars()  # 实例化阈值的滑动条
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # 边缘检测器
    kernel = np.ones((5, 5))#图像处理的卷积核
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # 图像处理：膨胀
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # 图像处理：腐蚀  膨胀腐蚀可以帮助我们消除缝隙和杂物

    ## 找到轮廓
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

    # 找到 最大轮廓
    biggest, maxArea = utlis.biggestContour(contours)  # #判断是否为最大矩形
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # 相当于切除不必要部分
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # 应用自适应阈值
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)#转换为灰度图像
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # 用于显示的图像阵列，即你要显示什么操作的结果，例如高斯模糊
        # ps 只有找到最大矩形时以下才会执行
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    else:#只有找不到最大矩形时以下才会执行
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # 标签显示
    lables = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]
    #堆叠窗口
    #stackedImage = utlis.stackImages(imageArray, 0.75, lables)#显示标签
    stackedImage = utlis.stackImages(imageArray, 0.75)#不显示标签

    cv2.imshow("Result", stackedImage)

    # 按下“s”键时保存图像
    if cv2.waitKey(1) & 0xFF == ord('s'):

        cv2.imwrite("Scanned/myImage_" + str(count) + ".jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
