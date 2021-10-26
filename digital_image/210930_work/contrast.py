# 测试通过对比度方法将图片分割开来
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def plot(grayHist):
    plt.plot(range(256), grayHist, 'r', linewidth=1.5, c='red')
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue]) # x和y的范围
    plt.xlabel("gray Level")
    plt.ylabel("Number of Pixels")
    plt.show()

if __name__ == "__main__":
    # 读取图像并转换为灰度图
    # img = cv2.imread('D:/data/G18678_2021-09-0111_18_19/10.0/18_5.jpeg')
    img = cv2.imread('D:/data/F60561-4_2021-06-10 09_00_41/10.0/5_6.jpeg')
    cv2.imshow("img",img)
    cv2.waitKey(0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 将图像转为灰度图像
    bw = 0
    shape = img_gray.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_gray[i][j] <= 120:
                bw = bw +1
    print("bw", bw)
    # 图像的灰度级范围是0~255
    hist_gray = plt.hist(img_gray.ravel(), bins=256, color='b')  # 显示灰度图像直方图
    plt.show()


