import os
import sys
import glob
import cv2

if __name__ == '__main__':
    # 获取图片路径
    pathImage = r'D:\Files\_datasets\Dataset-reference\VOCdevkit_Exposed_copper\VOC2007\SegmentationClass'
    ext = 'png'
    img_list = glob.glob(os.path.join(pathImage, '*.' + ext))
    #创建保存文件夹
    pathsaved = r'C:\Users\18493\Desktop\ppt2'
    if not os.path.exists(pathsaved):
        os.mkdir(pathsaved)
    # 读取图片
    print(img_list)
    for i,imgname in enumerate(img_list):

        img=cv2.imread(imgname)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度

        # 二值化
        # ret, imgbin = cv2.threshold(imgGray, 3, 255, cv2.THRESH_BINARY)
        ret, imgbin = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化


        #保存图片

        imgname = os.path.basename(imgname)
        cv2.imwrite(os.path.join(pathsaved, imgname.replace('_cutout','')), imgbin)
        print('save img successed:',os.path.join(pathsaved, imgname.replace('_cutout.png','.png')))

    # img = cv2.imread(img_list[0])
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
    # # 二值化
    # ret, imgbin = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow('imgbin',imgbin)
    # cv2.waitKey(0)
    # cv2.imwrite(pathsaved + f'/{imgname}'.replace('_cutout', ''), imgbin)