import pytesseract
from pytesseract import Output
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import imutils
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

#pip install fuzzywuzzy[speedup]

formNames = [
    "CSIO - Ontario Application for Automobile Insurance Garage Form (OAF 4) - ON1002e 201609", 
    "CSIO - Habitational Insurance Application CA2001e 201810",
    "CSIO - Automobile Policy Change Request CA1401e 200609",
    "CSIO - Automobile Loss Notice CA1501e 200609",
    "CSIO - Habitational Insurance Application CA2001e 201810",
    "CSIO - Ontario Application for Automobile Insurance ON1001e 201606"
]


def rotateImg():
    found = False
    formName = ""

    img = cv2.imread('img3.jpg')
    #count = 0

    # read the image and get the dimensions
    h, w, _ = img.shape # assumes color image

    for i in range(4):
        if(found == False):
            print("---------" + str(90 * i) + "---------")
            img = imutils.rotate_bound(img, 90)
            #cv2.imwrite(str(count) +'.jpg', img)
            #count += 1
            # Crop image to get bottom of image
            crop_position = int(img.shape[0]/2)
            cropped_img = img[img.shape[0] - crop_position:,:,:]

            # Extract text from the cropped image
            #print(pytesseract.image_to_string(cropped_img))
            identified_text = pytesseract.image_to_string(cropped_img).splitlines() 

            # Check if "All rights reserved" was found
            for word in identified_text:
                if('CSIO -' in word): 
                    print(word)
                    formName = word
                    #print(90 * i)
                    found = True
                    cv2.imwrite('upright.jpg', img)
                    break
        else:
            break
    if formName != "":
        return formName
    else: 
        return "Form Name not found"

def returnBestMatch():
    bestMatch = process.extractOne(rotateImg(), formNames)
    print(bestMatch)
    return bestMatch[0]


returnBestMatch()
