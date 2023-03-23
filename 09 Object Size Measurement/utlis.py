import cv2
import numpy as np
 
def getContours(img,cThr=[100,100],showCanny=False,minArea=1000,filter=0,draw =False):
    '''
    :param img: 输入图像
    :param cThr:边缘检测的阈值
    :param showCanny: 边缘显示与否
    :param minArea:最小面积
    :param filter:拟合点的过滤器，例如4代表只寻找四边形
    :param draw:过滤后的轮廓
    :return:
    '''
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)#膨胀
    imgThre = cv2.erode(imgDial,kernel,iterations=2)#腐蚀
    if showCanny:cv2.imshow('Canny',imgThre)
    contours,hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i,True)#计算图像轮廓的周长
            approx = cv2.approxPolyDP(i,0.02*peri,True)#一个连续光滑曲线折线化，对图像轮廓点进行多边形拟合
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx),area,approx,bbox,i])#finalCountours里的数据
            else:
                finalCountours.append([len(approx),area,approx,bbox,i])
    finalCountours = sorted(finalCountours,key = lambda x:x[1] ,reverse= True)#根据面积的大小进行升序
    if draw:#画出过滤后的轮廓
        for con in finalCountours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)
    return img, finalCountours
 
def reorder(myPoints):#将获得的顶点按‘z'顺序重新排序
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew
 
def warpImg (img,points,w,h,pad=20):
    # print(points)
    points =reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]#为了消除边界像素影响
    return imgWarp
 
def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

def measure(imgContours2,obj,scale):
    '''

    :param imgContours2: 轮廓列表
    :param obj: 遍历轮廓列表中的项
    :param scale: 放大倍率
    :return:
    '''
    cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
    nPoints = reorder(obj[2])
    nW = round((findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
    nH = round((findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)
    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                    (255, 0, 255), 3, 8, 0, 0.05)
    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                    (255, 0, 255), 3, 8, 0, 0.05)
    x, y, w, h = obj[3]
    cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                (255, 0, 255), 2)
    cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                (255, 0, 255), 2)