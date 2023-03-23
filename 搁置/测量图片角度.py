import cv2
import math


def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # size = len(pointsList)
        # if size % 3 == 0 and size != 0:
        #     cv2.line(img, tuple(pointsList[round((size - 1) / 3) * 3]), (x, y), (0, 0, 255), 2)#画线不成功
        cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
        pointsList.append([x, y])
        print(pointsList)


def getAngle(pointsList):
    point_1, point_2, point_3 = pointsList[-3:]  # 好聪明的办法，倒数三个
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a = math.sqrt((point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (
            point_2[1] - point_3[1]))
    b = math.sqrt((point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (
            point_1[1] - point_3[1]))
    c = math.sqrt((point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (
            point_1[1] - point_2[1]))
    A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
    # return B
    #print(int(B) + 1)
    cv2.putText(img,str(int(B) + 1),(point_2[0]-40,point_2[1]-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))


if __name__ == '__main__':
    path = 'test.jpg'
    img = cv2.imread(path)
    pointsList = []  # 保存坐标

    while True:  # 实时刷新图像
        if len(pointsList) % 3 == 0 and len(pointsList) != 0:
            # 这是为了可以再不刷新的情况下测量多个角度，故不是＝3
            getAngle(pointsList)
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', mousePoints)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 固定方法,记住,Q键是清除按钮
            pointsList.clear()  # 清空坐标表
            img = cv2.imread(path)
