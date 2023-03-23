import cv2
import numpy as np

''' 
原理：
cv找到出电容的轮廓，然后统计出其中的总像素cap_area
然后根据针孔的坐标找到针孔的轮廓，然后做同样的处理，统计出总像素target_area 
自动电容端面的标准面积reference_area
就可以获得针孔面积true_area=reference_area*(target_area /cap_area)



操作：
1.准备好缺陷电容图片
2.准备好缺陷的mask图片（labelme标注好后，实验voc的数据集转换脚本可生成在data_dataset_voc\SegmentationClass内）
3.改图片路径
'''


#获取轮廓
def get_contour(img_in,threshold1, threshold2,dilate_iterations=5,erode_iterations=3):
    '''

    :param img_in: 输入图像
    :param threshold1: canny第一个阈值
    :param threshold2: canny第二个阈值
    :param dilate_iterations:腐蚀重复次数
    :param erode_iterations:腐蚀重复次数
    :return:
    '''
    imgGray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
    imgThreshold = cv2.Canny(imgBlur, threshold1, threshold2)  # 边缘检测器
    kernel = np.ones((5, 5))  # 图像处理的卷积核
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=dilate_iterations)  # 图像处理：膨胀
    imgThreshold = cv2.erode(imgDial, kernel, iterations=erode_iterations)  # 图像处理：腐蚀  膨胀腐蚀可以帮助我们消除缝隙和杂物
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS

    return contours


def get_target_true_area(path,path2,reference_width= 1.3225,reference_height= 1.3225):
    '''

    :param path: 原图的路径
    :param path2: mask的路径
    :param reference_width: 参考物的宽
    :param reference_height: 参考物的长
    :return: 目标真实面积
    '''
    # 设定参照物尺寸 单位：mm
    # reference_width = 1.3225
    # reference_height = 1.3225
    reference_area = reference_width * reference_height  # 前提参照物是矩形
    # 读取图片和针孔mask （0.5缩放）
    img = cv2.imread(path)
    size_x, size_y = img.shape[0:2]
    img_in = cv2.resize(img, (int(size_y / 2), int(size_x / 2)))
    img2 = cv2.imread(path2)
    size_x, size_y = img2.shape[0:2]
    img2_in = cv2.resize(img2, (int(size_y / 2), int(size_x / 2)))
    # 找端面轮廓
    cap_contours = get_contour(img_in, 0, 25)
    img_out = cv2.drawContours(img_in, cap_contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS
    # 获得参照物的总像素
    cap_area = cv2.contourArea(cap_contours[0])
    # 获得参考比例
    reference_rate = cap_area / reference_area  # 或者分子分母反过来
    # 获得缺陷目标的总像素
    target_contours = get_contour(img2_in, 0, 15)
    target_area = cv2.contourArea(target_contours[0])
    # 计算出缺陷的真实面积
    true_area = target_area / reference_rate
    print('缺陷面积为：%.5f mm2' % true_area)
    return true_area


if __name__ == '__main__':
    path = r'D:\Files\test\Image\origin\E_pinhole_45.jpg'  # 电容图片
    path2 = r'D:\Files\test\Image\mask\E_pinhole_45.png'  # 缺陷mask图片   注意路径不能有中文


    get_target_true_area(path,path2)



