# import cv2
# import numpy as np
#
# path = r'test_imgs/1.jpg'
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# edges = cv2.Canny(img, 100, 200)
# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
#
# circles = np.uint16(np.around(circles))
#
# for i in circles[0, :]:
#     center = (i[0], i[1])
#     radius = i[2]
#     # 绘制圆
#     cv2.circle(img, center, radius, (0, 255, 0), 2)
#     # # 测量直径
#     # diameter = 2 * radius
#     # cv2.putText(img, f"{diameter}", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
# cv2.imshow("detected circles", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2 
import matplotlib.pyplot as plt
import numpy as np

src = cv2.imread(r"test_imgs/3.jpg")
img = src.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯
imgBlur = cv2.GaussianBlur(img_gray, (5, 5), 2)
# 进行中值滤波
dst_img = cv2.medianBlur(imgBlur, 7)

# 霍夫圆检测
circle = cv2.HoughCircles(dst_img, cv2.HOUGH_GRADIENT, 1, 100,
                         param1=100, param2=100, minRadius=0, maxRadius=10000)

print(circle)
# 将检测结果绘制在图像上
for i in circle[0, :]:  # 遍历矩阵的每一行的数据
    # 绘制圆形
    cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 10)
    # 绘制圆心
    cv2.circle(img, (int(i[0]), int(i[1])), 10, (255, 0, 0), -1)
    print('圆心坐标：', i[0], i[1])
    print('圆半径：', i[2])

# 显示图像
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 8), dpi=100)
axes[0].imshow(src[:, :, ::-1])
axes[0].set_title("原图")
axes[1].imshow(img[:, :, ::-1])
axes[1].set_title("霍夫圆检测后的图像")
axes[2].imshow(imgBlur, cmap="gray")
axes[2].set_title("高斯滤波后的图像")
axes[3].imshow(dst_img, cmap="gray")
axes[3].set_title("中值滤波后的图像")
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.show()
