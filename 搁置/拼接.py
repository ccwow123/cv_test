# -*- coding: utf-8 -*-
import cv2
import os

mainFolders = 'Image'
myFolders = os.listdir(mainFolders)
print(myFolders)

for folder in myFolders:
    path = mainFolders + '/' + folder
    images = []
    myList = os.listdir(path)
    print(f'本文件夹包含：{len(myList)}个图像')
    for imgN in myList:
        curImg = cv2.imread(f'{path}/{imgN}')
        curimg = cv2.resize(curImg, (0, 0), fx=0.2, fy=0.2)  # 看https://www.jianshu.com/p/0deabe02a379
        images.append(curImg)

        #拼接操作
        stitcher= cv2.Stitcher_create()
        (status, result) = stitcher.stitch(images)
        if status == cv2.STITCHER_OK:
            print(' Panorama Generated')
            cv2.imshow(folder,result)
            cv2.waitKey(1)
        else:
            print('Panorama Generation Unsuccessful')

cv2.waitKey(0)
