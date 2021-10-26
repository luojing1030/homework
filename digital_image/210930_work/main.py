import cv2
import matplotlib.pyplot as plt


import numpy as np
import math
# cv2.imshow("night", img)
img = cv2.imread("C:/Users/admin/Desktop/test_picture.jpg")

# 定义伽马变换函数
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)


img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 将图像转为灰度图像
img_size = img_gray.shape    # 获取图像大小
total_pix = img_size[0] * img_size[1]   # 计算图像总的像素个数
hist_gray = plt.hist(img_gray.ravel(), bins=256, color='b')  # 显示灰度图像直方图
plt.show()
total_gray = 0   # 用来存储整个图像的像素和
for i in range(256):  # 根据直方图计算像素和
    total_gray = total_gray + i * hist_gray[0][i]

  # 计算图像像素平均值
mean_value = total_gray / (total_pix * 255) # 求出每个像素均值，再除以255转到0-1区间
print("mean_value", mean_value)
gamma = 1
if mean_value < 0.5:  # gamma变换指数值默认为1，若图像灰度归一化后均值小于0.5，则计算指数值，将通过gamma变换将均值变为0.5
    gamma = math.log(10, 0.5) / math.log(10, mean_value)  # 计算将均值变换到0.5，伽马变换所需的指数
print("gamma", 1/gamma)
img_gamma = adjust_gamma(img_gray, gamma)   # 进行伽马变换
hist_gamma = plt.hist(img_gamma.ravel(), bins=255, color='y')  # 伽马变换后的直方图
plt.show()
cv2.imshow("img_gray", img_gray)
cv2.imshow("img_gamma", img_gamma)

cv2.waitKey(0)
