from __future__ import print_function

# import image_slicer
import math

# import argparse
import cv2
import numpy as np
import pandas as pd
# from imutils import perspective
# from imutils import contours
from PIL import Image
from scipy.signal import convolve2d
from sklearn.cluster import KMeans


class cPoints:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.P = -1

    def assignP(self, Pin):
        self.P = Pin


# Corner Detection Algorithm
def detect_corner(Img):
    img = cv2.imread(Img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=10, useHarrisDetector=True)
    corners = np.int0(corners)
    # print(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    cv2.imwrite('GFeature.png',img)
    # plt.imshow(img),plt.show()
    return corners


# Noise Estimation
def estimate_noise(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    H, W = I.shape

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))

    return sigma
 

# Line Detection Function
def lineDetectFun(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 5
    maxLineGap = 100
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=minLineLength,
                            maxLineGap=maxLineGap)

    for line in lines:
        # print(line)
        for x1, y1, x2, y2 in line:
            # need to process each datapoint here find the important lines with key values in every coordinates
            cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)

    cv2.imwrite('houghlines5.png', img)


def convertFromListToArray(listO):
    dummyList = []
    for i in listO:
        dummyList.append(i[0])

    return np.array([np.array(xi) for xi in dummyList])


# Loop over each contour for size
def findContours(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # find major contours in the image
    cnts, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(cnts)):
        hull = cv2.convexHull(cnts[i])
        hull_list.append(hull)

    maxArea = -1
    valid_Draw = []
    for c in hull_list:
        if cv2.contourArea(c) < maxArea:
            continue
        maxArea = cv2.contourArea(c)
        valid_cnts = c
        valid_Draw.clear()
        valid_Draw.append(c)

    # Draw contours + hull results
    for i in range(len(valid_Draw)):
        cv2.drawContours(image, valid_Draw, i, (255, 153, 255))

    cv2.imwrite("resultContour.jpg", image)
    Result = convertFromListToArray(valid_cnts)

    image = cv2.fillPoly(image, pts=[np.array(valid_cnts)], color=(255, 255, 255))
    stencil = np.zeros(image.shape).astype(image.dtype)

    color = [255, 255, 255]
    cv2.fillPoly(stencil, [np.array(valid_cnts)], color)
    resultimage = cv2.bitwise_and(image, stencil)
    cv2.imwrite("result.jpg", resultimage)

    # convert the points to the right format

    # draw the contour and show it

    return Result


# Edge Detection Function
def edgeDetectFun(Images, template, cv2Img=False):
    if not template and not cv2Img:
        img = cv2.imread(Images)
    else:
        img = Images
    sigma = estimate_noise(img)
    print(sigma)
    if sigma > 4:
        #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, None)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, None)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, None)

        img = cv2.dilate(img, None, iterations=1)
        img = cv2.erode(img, None, iterations=1)

        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)

    edges = cv2.Canny(img, 20, 30)
    edges = cv2.erode(edges, None, iterations=1)
    edges = cv2.dilate(edges, None, iterations=1)

    cv2.imwrite('edgeTest.png', edges)

    lineDetectFun('edgeTest.png')

    contourCoordinates = findContours('houghlines5.png')

    return contourCoordinates


# Processing Function
def cornerIdentify(points):
    # find the corresponding corners for the elements
    points = sorted(points, key=sum)
    print(points)

    tl = points[0]
    br = points[3]

    if points[1][0] > points[2][0]:
        tr = points[1]
        bl = points[2]
    elif points[1][0] < points[2][0]:
        tr = points[2]
        bl = points[1]
    elif points[1][1] > points[2][1]:
        # if for whatever reason, x value is the same, which is highly unlikely
        # compare y value
        tr = points[2]
        br = points[1]
    else:
        tr = points[1]
        br = points[2]
    return tl, tr, bl, br


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def adjustmentfunc(tl, tr, bl, br):
    top = [tr[0] - tl[0], tr[1] - tl[1]]
    right = [br[0] - tr[0], br[1] - tr[1]]
    bottom = [bl[0] - br[0], bl[1] - br[1]]
    left = [tl[0] - bl[0], tl[1] - bl[1]]

    trAngle = angle(left, top)
    tlAngle = angle(top, right)
    brAngle = angle(right, bottom)
    blAngle = angle(bottom, left)

    print("angles")
    print(trAngle)
    print(tlAngle)
    print(brAngle)
    print(blAngle)

    print(float(brAngle - 1.57))
    print(float(1.57 - blAngle))
    print(float(tlAngle - 1.57))
    print(float(trAngle - 1.57))

    # find a logic to check for offset points
    if tlAngle - 1.57 > 0.03 and 1.57 - trAngle > 0.03 and blAngle - 1.57 < 0.03 and brAngle - 1.57 < 0.04:
        print('here')
        # change the point
        tl = [tl[0], tr[1]]
    # more adjustment needed
    if trAngle - 1.57 > 0.03 and 1.57 - tlAngle > 0.03 and blAngle - 1.57 < 0.03 and brAngle - 1.57 < 0.03:
        # change the point
        tr = [tr[0], tl[1]]
    if blAngle - 1.57 > 0.03 and 1.57 - brAngle > 0.03 and tlAngle - 1.57 < 0.03 and trAngle - 1.57 < 0.03:
        print('here1')
        # change the point
        br = [br[0], bl[1]]
    if brAngle - 1.57 > 0.01 and 1.57 - blAngle > 0 and tlAngle - 1.57 < 0.05 and trAngle - 1.57 < 0.03:
        print('here2')
        # change the point
        bl = [tl[0], br[1]]

    return tl, tr, bl, br


def crop_func(filename, points, template=False, cv2Img=False):
    if not template and not cv2Img:
        img = cv2.imread(filename)
    else:
        img = filename

    rows, cols, ch = img.shape

    tl, tr, bl, br = cornerIdentify(points)

    distw = math.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)

    distl = math.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)

    print(distw)
    print(distl)

    # adjustment happen here 
    print("check")
    print(tl)
    print(tr)
    print(bl)
    print(br)

    tl, tr, bl, br = adjustmentfunc(tl, tr, bl, br)
    pts1 = np.float32([tl, tr, bl, br])

    print(tl)
    print(tr)
    print(bl)
    print(br)

    if distw > distl:
        # this is landscape
        print("landscape")
        pts2 = np.float32([[0, 0], [4000, 0], [0, 3000], [4000, 3000]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (4000, 3000))
        cv2.imwrite("finalResult.jpg", dst)

        colorImage = Image.open("finalResult.jpg")
        transposed = colorImage.transpose(Image.ROTATE_90)
        transposed.save("finalResult.jpg")


    else:
        pts2 = np.float32([[0, 0], [3000, 0], [0, 4000], [3000, 4000]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (3000, 4000))
        cv2.imwrite("finalResult2.jpg", dst)

    # return tl, tr, bl, br
    return dst


def crop_template_func(img, points):
    height, width, channel = img.shape

    print(height)
    print(width)

    tl, tr, bl, br = cornerIdentify(points)

    topMargin = min(tl[1], tr[1])
    bottomMargin = min((height - bl[1]), (height - br[1]))
    leftMargin = min(tl[0], bl[0])
    rightMargin = min((width - br[0]), (width - tr[0]))

    return (leftMargin / width), (topMargin / height), (bottomMargin / height), (rightMargin / width)


# Corner Points Selection
def find_min_max_points(points):
    xValue = []
    yValue = []
    for i in points:
        xValue.append(i[0])
        yValue.append(i[1])

    Data = {
        'x': xValue,
        'y': yValue
    }
    # convert into pandas
    df = pd.DataFrame(Data, columns=['x', 'y'])

    kmeans = KMeans(n_clusters=4).fit(df)
    centroids = kmeans.cluster_centers_
    print(centroids)
    print(kmeans.labels_)

    # figure what each index in cntroids represents which element out of the 4
    tlv, trv, blv, brv = cornerIdentify(centroids)
    for i, val in enumerate(centroids):
        if (val == tlv).all():
            tl = i
        elif (val == trv).all():
            tr = i
        elif (val == brv).all():
            br = i
        elif (val == blv).all():
            bl = i

    plabel = kmeans.labels_
    trl = []
    tll = []
    brl = []
    bll = []

    # separate all the points into different clusters
    for i, val in enumerate(plabel):
        if val == tl:
            tll.append(points[i])
        elif val == tr:
            trl.append(points[i])
        elif val == br:
            brl.append(points[i])
        elif val == bl:
            bll.append(points[i])

    # find tl and br

    tlPoint = sorted(tll, key=sum)[0]
    brPoint = sorted(brl, key=sum)[-1]

    # find the tr and bl by checking the surrounding colors

    trPoint = amb_detect_corner(trl)
    blPoint = amb_detect_corner(bll)

    finalList = [[tlPoint[0], tlPoint[1]], [brPoint[0], brPoint[1]], trPoint, blPoint]

    return finalList


def amb_detect_corner(ls):
    im = Image.open('result.jpg')
    img = cv2.imread('result.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_im = im.convert('RGB')

    # for all the points in each corner find the surrounding pixels color

    bestCorner = cPoints(-1, -1)

    for i in ls:

        newP = cPoints(i[0], i[1])
        prob = 0

        # check the surrounding

        xc = -2
        counter = 0
        while xc < 3:
            yc = -2
            while yc < 3:
                counter = counter + 1

                if (int(i[0] + xc)) < 0:
                    inX = 0
                else:
                    inX = int(i[0] + xc)

                if (int(i[1] + yc)) < 0:
                    iny = 0
                else:
                    iny = int(i[1] + yc)
                r, g, b = rgb_im.getpixel((inX, iny))
                sumT = r + g + b
                if sumT < 20:
                    prob = prob + 1
                yc = yc + 1
            xc = xc + 1

        prob = prob / counter
        newP.assignP(prob)

        if prob > bestCorner.P:
            bestCorner = newP

        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 200, -1)
    return [bestCorner.x, bestCorner.y]


def main_process_fun(filename, templateFlag, cv2Img=False):
    contourPoints = edgeDetectFun(filename, templateFlag, cv2Img)
    points = find_min_max_points(contourPoints)
    if templateFlag:
        return crop_template_func(filename, points)
    else:
        return crop_func(filename, points, cv2Img)


main_process_fun('2.jpg', False)
