from concurrent.futures import ThreadPoolExecutor
import concurrent.futures as pool
from utils2 import *
import time
'''
多线程版本
切矩形图片
可以使用二值和高斯两种方法
不可以查看保存了多少图片
可能会丢图片
2.0先高斯再二值好用
method = 'bin' 使用手动二值
method = 'bin2' 使用自动二值
method = 'gauss' 使用高斯
'''
def edge_cut(img_list):
    pathImage = r'C:\Users\18493\Desktop\test\images\cam1.jpg'
    method = 'bin2'
    # 1、读取图片
    img = cv2.imread(os.path.join(pathImage, img_list))
    ori_imgHeight, ori_imgWidth, _ = img.shape
    # img = cv2.resize(img, (ori_imgWidth, ori_imgHeight))  # RESIZE IMAGE
    # 2、图像处理
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
    # 2.1 二值和高斯两种方法
    if method == 'bin':
#！！！修改处1
        ret, imgbin = cv2.threshold(imgGray, 1, 255, cv2.THRESH_BINARY)  # 二值化
    elif method == 'bin2':
        ret, imgbin = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
    elif method == 'gauss':
        imgbin = imgBlur
# ！！！修改处2
    imgCanny = cv2.Canny(imgbin, 1, 1)  # 边缘检测器

    kernel = np.ones((5, 5))  # 图像处理的卷积核
# ！！！修改处3
    imgDial = cv2.dilate(imgCanny, kernel, iterations=16)  # 图像处理：膨胀
    imgCanny = cv2.erode(imgDial, kernel, iterations=0)  # 图像处理：腐蚀  膨胀腐蚀可以帮助我们消除缝隙和杂物

    # 3、找到轮廓
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

    # 3.1、 找到轮廓中的最大轮廓
# ！！！修改处4
    biggest, maxArea = biggestContour(contours,500,6)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [ori_imgWidth, 0], [0, ori_imgHeight], [ori_imgWidth, ori_imgHeight]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (ori_imgWidth, ori_imgHeight))

        # 相当于切除不必要部分
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (ori_imgWidth, ori_imgHeight))
    # 4、保存图片
    mkdir(os.path.join(pathImage, "Scanned"))  # 创建保存目录
    c_img = os.path.basename(img_list)
    ret=cv2.imwrite(os.path.join(pathImage, "Scanned", c_img), imgWarpColored)
    if ret==True:
        print(f'{c_img}保存成功')
    else:
        print(f'{c_img}保    存   失   ！！')

    # # 打开文件夹
    # start_directory = os.path.join(pathImage, "Scanned")
    # os.system("explorer.exe %s" % start_directory)
if __name__ == '__main__':
    pathImage = r'C:\Users\18493\Desktop\cam_0'
    img_list = getFileList(pathImage)
    #多线程
    time_start = time.time()
    with ThreadPoolExecutor() as executor:
        executor.map(edge_cut,img_list)
    time_end = time.time()
    print('totally cost', time_end - time_start)

    # #多进程
    # t1=time.time()
    # with pool.ProcessPoolExecutor() as executor2:
    #     executor2.map(edge_cut,img_list)
    # t2=time.time()
    # print('mulit花费时间：',t2-t1)

    # 打开文件夹
    start_directory = os.path.join(pathImage, "Scanned")
    os.system("explorer.exe %s" % start_directory)




