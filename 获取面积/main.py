import cv2
import numpy as np
import math
''' 
cv找到出电容的轮廓，然后统计出其中的总像素cap_area
然后根据针孔的坐标找到针孔的轮廓，然后做同样的处理，统计出总像素target_area (怎么办？)
自动电容端面的标准面积reference_area
就可以获得针孔面积true_area=reference_area*(target_area /cap_area)
'''
def get_target_area(self, reference_area)
    pass
if __name__ == '__main__':
    path = r'D:\Files\MyData\images\E_pinhole_451.jpg'
    #设定参照物尺寸
    reference_width=1.3225
    reference_height=1.3225
    reference_area=reference_width*reference_height#前提参照物是矩形
    #读取图片
    img=cv2.imread(path)
    size_x, size_y = img.shape[0:2]
    img_in = cv2.resize(img, (int(size_y / 2), int(size_x / 2)))
    #找轮廓
    imgGray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
    imgThreshold = cv2.Canny(imgBlur, 0, 35)  # 边缘检测器
    kernel = np.ones((5, 5))  # 图像处理的卷积核
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=5)  # 图像处理：膨胀
    imgThreshold = cv2.erode(imgDial, kernel, iterations=3)  # 图像处理：腐蚀  膨胀腐蚀可以帮助我们消除缝隙和杂物
    imgContours = img_in.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img_in.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    img_out=cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS


    #2.获得参照物的总像素
    cap_area=cv2.contourArea(contours[0])
        #获得参考比例
    reference_rate=cap_area/reference_area#或者分子分母反过来

    #3.求目标的面积？？？

    target_area=get_target_area(target_points)

    true_area=target_area/reference_rate

    print('缺陷面积为：%.4f mm2' % true_area)

    cv2.imshow('drawimg', img_out)
    cv2.waitKey(0)

