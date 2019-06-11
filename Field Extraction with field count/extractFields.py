'''
https://medium.com/coinmonks/a-box-detection-algorithm-for-any-image-containing-boxes-756c15d7ed26?fbclid=IwAR1eu-Abnulgrqac74K7XFpDCp1116u1V7VZJWh2SJMzTh4cbIYX69q4KU4
'''

import cv2
import numpy as np
import argparse
import imutils
import pdfrw
import os
 


# ----------------------------------------------------------------
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# crops and stretches an image to be rectangular based on quadrilateral of points
# inputs: cv2 image
# inputs: four points that you want to be the new corners of the image
# outputs: cropped/stretched cv2 image
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped



# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# orders corners, taken from online
# inputs: array of points
# outputs: array of points (I think, don't worry about this one I stole it from online)
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

# https://stackoverflow.com/questions/4195453/how-to-resize-an-image-with-opencv2-0-and-python2-6
# shrinks a cv2 image to max 1000 px wide or high
# inputs: cv2 image
# outputs: cv2 image
def shrinkImg(img):
    height, width = img.shape[:2]
    max_height = 1000
    max_width = 1000

    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return img

# TODO: Test more
# takes an image, grayscales it, finds the corners of the form, and crops/stretches it to those corners
# inputs: cv2 image
# inputs: skip_corners (optional): skips the corner detection
# outputs: cv2 image
def fitToForm(img, skip_corners = False):
    img = shrinkImg(img)
    if (len(img.shape) == 3):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if skip_corners:
        return scaleImg(gray)

    height = len(gray)
    width = len(gray[0])

    topLeft = [width, height]
    topRight = [0, height]
    bottomLeft = [width, 0]
    bottomRight = [0, 0]

    error_distance = 3

    # TODO: Make this less hacky
    cropX = int(width*0.03)
    cropY = int(height*0.01)

    for y in range(cropY, height - error_distance - cropY):
        for x in range(cropX, width - error_distance - cropX):
            if np.mean([gray[i][x:x+error_distance] for i in range(y, y+error_distance)]) < 150:
                if x + y < topLeft[0] + topLeft[1]:
                    topLeft = [x, y]
                if x - y > topRight[0] - topRight[1]:
                    topRight = [x, y]
                if x - y < bottomLeft[0] - bottomLeft[1]:
                    bottomLeft = [x, y]
                if x + y > bottomRight[0] + bottomRight[1]:
                    bottomRight = [x, y]

    crop = four_point_transform(gray, np.array([topLeft, topRight, bottomLeft, bottomRight]))

    return scaleImg(crop)



# scales an image to 600px wide
# inputs: cv2 image
# outputs: cv2 image
def scaleImg(img):
    height, width = img.shape[:2]
    new_width = 3000

    # get scaling factor
    scaling_factor = new_width / float(width)
    # resize image
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return img

# inputs: pdf_path: path to the template pdf
# inputs: img_paths: array of cv2 images for each page in the pdf (ordered)
# inputs: skip_corners (optional): skips the automatic corner detection and just uses the full image if true
# outputs: dict of bounding boxes see /csio-forms/oaf1.json for an example
def getPdfBoxes(pdf_path, img_paths, skip_corners=False):
    ANNOT_KEY = '/Annots'
    ANNOT_FIELD_KEY = '/T'
    ANNOT_VAL_KEY = '/V'
    ANNOT_RECT_KEY = '/Rect'
    SUBTYPE_KEY = '/Subtype'
    WIDGET_SUBTYPE_KEY = '/Widget'
    PARENT_KEY = '/Parent'
    FIELD_TYPE_KEY = '/FT'
    CHECKBOX_KEY = '/Btn'
    BOX_KEY = '/Rect'
    SIZE_KEY = '/Size'

    bounding_boxes = []

    template_pdf = pdfrw.PdfReader(pdf_path)
    template_pdf.Root.AcroForm.update(pdfrw.PdfDict(NeedAppearances=pdfrw.PdfObject('true')))

    # IDK what this is think it's width
    width = template_pdf[SIZE_KEY]

    for i in range(len(template_pdf.pages)):
        bounding_boxes.append({})
        annotations = template_pdf.pages[i][ANNOT_KEY]

        for annotation in annotations:
            if annotation[SUBTYPE_KEY] == WIDGET_SUBTYPE_KEY:
                if annotation[PARENT_KEY] and annotation[PARENT_KEY][ANNOT_FIELD_KEY]:
                    if annotation[PARENT_KEY][ANNOT_FIELD_KEY][1:-1] not in bounding_boxes[i].keys():
                        bounding_boxes[i][annotation[PARENT_KEY][ANNOT_FIELD_KEY][1:-1]] = []

                    box = []

                    type = 'checkbox' if annotation[PARENT_KEY][FIELD_TYPE_KEY] == CHECKBOX_KEY else 'text'

                    for point in annotation[BOX_KEY]:
                        box.append(float(point) / float(width))

                    bounding_boxes[i][annotation[PARENT_KEY][ANNOT_FIELD_KEY][1:-1]].append({'box': box, 'type': type})

                if annotation[ANNOT_FIELD_KEY]:
                    if annotation[ANNOT_FIELD_KEY][1:-1] not in bounding_boxes[i].keys():
                        bounding_boxes[i][annotation[ANNOT_FIELD_KEY][1:-1]] = []

                    box = []

                    type = 'checkbox' if annotation[FIELD_TYPE_KEY] == CHECKBOX_KEY else 'text'

                    for point in annotation[BOX_KEY]:
                        box.append(float(point) / float(width))

                    bounding_boxes[i][annotation[ANNOT_FIELD_KEY][1:-1]].append({'box': box, 'type': type})

    scale = 3.92
    offset_x = 0.009
    offset_y = 0.016

    for i in range(len(img_paths)):
        form = fitToForm(cv2.imread(img_paths[i]), skip_corners=skip_corners)
        cv2.imwrite("test1.jpg", form)
        width = len(form[0])
        height = len(form)
        for key in bounding_boxes[i].keys():
            for j, box in enumerate(bounding_boxes[i][key]):
                if key == 'Reset':
                    continue

                box['box'][1], box['box'][3] = box['box'][3], box['box'][1]

                box['box'][0] = max((box['box'][0] + offset_x) * scale, 0)
                box['box'][1] = max(height / width - (box['box'][1] + offset_y) * scale, 0)
                box['box'][2] = min((box['box'][2] + offset_x) * scale, 1)
                box['box'][3] = min(height / width - (box['box'][3] + offset_y) * scale, height / width)

                left_x = box['box'][0] * width
                right_x = box['box'][2] * width
                top_y = box['box'][1] * width
                bot_y = box['box'][3] * width

                bounding_boxes[i][key][j]['box'] = [left_x, top_y, right_x, bot_y]
    return bounding_boxes


#Given bounds of all input fields, this function returns the number of fields enclosed by 
#a set of coordinates
#Inputs:
#    fieldBounds: Tuples of all input field coordinates
#    coordsToCheck: Array of coordinates to test
#Outputs:
#    count: Number of input fields enclosed by test coordinates
#
def numFields(fieldBounds, coordsToCheck):
    count = 0
    padding = 15
    left_x = coordsToCheck[0] - padding
    right_x = coordsToCheck[0] + coordsToCheck[2] + padding
    top_y = coordsToCheck[1] - padding
    bot_y = coordsToCheck[1] + coordsToCheck[3] + padding
    print(coordsToCheck)
    for key in fieldBounds.keys():
    	for j, box in enumerate(fieldBounds[key]):
    		if(left_x <= fieldBounds[key][j]['box'][0] and 
				right_x >= (fieldBounds[key][j]['box'][2]) and
				top_y <= fieldBounds[key][j]['box'][1] and
				bot_y >= (fieldBounds[key][j]['box'][3])):
    			count += 1
    print(count)
    return count




def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def deleteExtraImages(path):
    croppedImages = os.listdir(path)

    for item in croppedImages:
        if item.endswith("-0.jpg"):
            os.remove(os.path.join(path, item))

# Read the image
img = scaleImg(cv2.imread('./oafpbm/2.jpg', 0))


cv2.imwrite('new.jpg', img)
 
# Thresholding the image
(thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY|     cv2.THRESH_OTSU)
# Invert the image
img_bin = 255-img_bin 
cv2.imwrite("Image_bin.jpg",img_bin)

# Defining a kernel length
kernel_length = np.array(img).shape[1]//140
 
# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
# A kernel of (3 X 3) ones.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Morphological operation to detect vertical lines from an image
img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
# Morphological operation to detect horizontal lines from an image
img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
alpha = 0.5
beta = 1.0 - alpha
# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=3)
(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("img_final_bin.jpg",img_final_bin)

# Find contours for image, which will detect all the boxes
contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort all the contours by top to bottom.
(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

print("here")

dirpath = os.getcwd()
print(dirpath +'/oafpbm')

fieldBounds = getPdfBoxes('OAF.pdf', ['oafpbm/' + i for i in os.listdir(dirpath + '/oafpbm/') if i[len(i)-4:] == '.jpg'], skip_corners=True)

idx = 0
for c in contours:
    # Returns the location and width,height for every contour
	x, y, w, h = cv2.boundingRect(c)
    # If the box height is greater then 20, width is >80, then only save it as a box in "cropped/" folder.
	if (w > 80 and h > 20) and w > 3*h:
		print(fieldBounds[1])
		numInputs = numFields(fieldBounds[1], [x,y,w,h])
		idx += 1
		new_img = img[y:y+h, x:x+w]
		cv2.imwrite('./Cropped/' + str(idx) + '--' + str(numInputs) + '.jpg', new_img)

deleteExtraImages(dirpath + "/Cropped/")







