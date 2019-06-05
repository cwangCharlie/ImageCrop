'''
    This script takes a perspective image and 
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

#img = cv2.imread('img1.jpg')
img = cv2.imread('img2.jpg')

rows,cols,ch = img.shape

# Image 1 regular
#pts1= np.float32([[550,715], [2640,725],[0, 3775],[3024,3800]])
# Image 2 rotated right
pts1 = np.float32([[3242, 535],[3250, 2520], [214,0],[213,3024]])


pts2 = np.float32([[0,0],[3000,0],[0,4000],[3000,4000]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (3000,4000))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()