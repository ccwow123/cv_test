import cv2
import utlis
 
###################################
webcam = False
path = r'09 Object Size Measurement/1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)#第一参数为标号，具体看https://blog.csdn.net/weixin_47965042/article/details/113359922
cap.set(3,1920)
cap.set(4,1080)
scale = 3
wP = 210 *scale
hP= 297 *scale
###################################
while True:
    if webcam:success,img = cap.read()#其中success是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False
    else: img = cv2.imread(path)
 
    imgContours , conts = utlis.getContours(img,minArea=5,filter=4)
    if len(conts) != 0:#为了判断最大轮廓中的内容是否是空的，我们需要内容
        biggest = conts[0][2]#轮廓列表中最大的轮廓。即第0个
        #print(biggest)
        imgWarp = utlis.warpImg(img, biggest, wP,hP)
        ###    内容矩形的测量   如果要其他功能就修改这里
        imgContours2, conts2 = utlis.getContours(imgWarp,minArea=200, filter=4,cThr=[50,50],draw = False)
        if len(conts2) != 0:
            for obj in conts2:
                utlis.measure(imgContours2,obj,scale)
        cv2.imshow('A4', imgContours2)
 
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    cv2.waitKey(1)