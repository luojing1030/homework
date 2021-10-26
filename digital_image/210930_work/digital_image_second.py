# 设计一种图像线性滤波加速算法，并编程实现。（实例，加速前后速度对比）；代码 + 测试图例
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def sp_noise(image, prob):
  """
  添加椒盐噪声
  :param image:
  :param prob: 噪声比例
  :return: image with sp_noise
  """
  output = np.zeros(image.shape, np.uint8)
  thres = 1 - prob
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rand_num = random.random()
      if rand_num < prob:
        output[i][j] = 0
      elif rand_num > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output

def numpy_to_gray(array):
  array = Image.fromarray(array)
  array = array.convert('L')
  return array

img = cv2.imread("C:/Users/Admin/Desktop/water.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


f = sp_noise(img_gray, 0.0025)
size = f.shape
f = np.array(f, dtype=float)
way1 = f.copy()
way2 = f.copy()
start1 = time.time()
for i in range(1, size[0]-1, 1):
  for j in range(1, size[1]-1, 1):
    way1[i][j] = (f[i][j] + f[i-1][j-1] + f[i-1][j] + f[i-1][j+1] + f[i][j-1] + f[i][j+1]
                  + f[i+1][j-1] + f[i+1][j] + f[i+1][j+1]) / 9
time1 = time.time() - start1
start2 = time.time()
for i in range(1, size[0]-1, 1):
  for j in range(1, size[1]-1, 1):
    if j == 1:
      c1 = (f[i-1][j-1] + f[i][j-1] + f[i+1][j-1]) / 9
      c2 = (f[i-1][j] + f[i][j] + f[i+1][j]) / 9
      c3 = (f[i+1][j+1] + f[i][j+1] + f[i-1][j+1]) / 9
    else:
      c1 = c2
      c2 = c3
      c3 = (f[i+1][j+1] + f[i][j+1] + f[i-1][j+1]) / 9
    way2[i][j]= c1 + c2 + c3
time2 = time.time() - start2
f = f.astype(int)
f = numpy_to_gray(f)
f.save("C:/Users/Admin/Desktop/f.jpg")
way1 = way1.astype(int)
way1 = numpy_to_gray(way1)
way1.save("C:/Users/Admin/Desktop/way1.jpg")
way2 = way2.astype(int)
way2 = numpy_to_gray(way2)
way2.save("C:/Users/Admin/Desktop/way2.jpg")

print("time1:", time1)
print("time2:", time2)
start = time.time()
img_mean = cv2.blur(img_gray, (3, 3))
way = time.time() - start
print("way",way)



