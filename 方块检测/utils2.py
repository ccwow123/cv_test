import glob
import cv2
import numpy as np
import os
## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)#行数
    cols = len(imgArray[0])#列数
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                #文字的背景
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, str(lables[d][c]), (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (255, 0, 255), 2)
    return ver

# 重新排序
def reorder(myPoints):
    '''
    坐标为
    0   1
    2   3
    '''
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggestContour(contours,maxArea,epsilon):
    biggest = np.array([])
    max_area = 0
    epsilon=epsilon/100
    for i in contours:  # 判断是否为矩形，且要求要有一定大小
        area = cv2.contourArea(i)
        if area > maxArea:  # 调整矩形大小
            peri = cv2.arcLength(i, True)#轮廓周长
            approx = cv2.approxPolyDP(i, epsilon * peri, True)#第二个参数叫epsilon,是从原始轮廓到近似轮廓的最大距离,是一个准确率参数
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


def nothing(x):
    pass


def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 500, 300)
    cv2.createTrackbar("Th_bin1", "Trackbars", 34, 255, nothing)
    cv2.createTrackbar("Th_canny1", "Trackbars", 1, 255, nothing)
    cv2.createTrackbar("Th_canny2", "Trackbars", 1, 255, nothing)
    cv2.createTrackbar("pengzhang", "Trackbars", 1, 15, nothing)
    cv2.createTrackbar("fushi", "Trackbars", 1, 15, nothing)
    cv2.createTrackbar("maxArea", "Trackbars", 500, 1500, nothing)
    cv2.createTrackbar("area_epsilon", "Trackbars", 6, 10, nothing)


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Th_bin1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Th_canny1", "Trackbars")
    Threshold3 = cv2.getTrackbarPos("Th_canny2", "Trackbars")
    Threshold4 = cv2.getTrackbarPos("pengzhang", "Trackbars")
    Threshold5 = cv2.getTrackbarPos("fushi", "Trackbars")
    Threshold6 = cv2.getTrackbarPos("maxArea", "Trackbars")
    Threshold7 = cv2.getTrackbarPos("area_epsilon", "Trackbars")

    src = Threshold1, Threshold2, Threshold3, Threshold4, Threshold5, Threshold6, Threshold7
    return src
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录,创建目录操作函数
        '''
        os.mkdir(path)与os.makedirs(path)的区别是,当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录
        '''
        # 此处路径最好使用utf-8解码，否则在磁盘中可能会出现乱码的情况
        # os.makedirs(path.decode('utf-8'))
        os.makedirs(path)
        # print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False

def getFileList(path):
    """
       获得所有文件及子文件夹的列表，使用的时候最后只含单文件
       输入 path:文件夹根目录
       返回： 文件名列表
       例子：
        imglist = getFileList(path)
    """
    paths=os.listdir(path)
    imglist=paths
    print('本次执行检索到 ' + str(len(imglist)) + ' 个文件\n')
    return imglist

def getFileList2(path,ext):
    """
       获取文件夹及其子文件夹中文件列表
       输入 path：文件夹根目录
       输入 ext: 扩展名
       返回： 文件路径列表
       例子：
        imglist = getFileList(path, 'jpg')
        print('本次执行检索到 '+str(len(imglist))+' 张图像\n')
       """
    WSI_MASK_PATH = path #存放图片的文件夹路径
    paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.{}'.format(ext)))
    imglist=paths
    print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')
    return imglist
