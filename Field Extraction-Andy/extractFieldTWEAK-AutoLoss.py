#todo: also need to check how many fields are in each cut out, in the case we pad and it goes into another field

import cv2
import numpy as np
import argparse
import imutils
import pdfrw
import os
import asyncio
import base64
from aiohttp import ClientSession
import json

google_vision_api_key = ('AIzaSyCujojNEZTGf4pak-5TKTnQRrEP5KO7JAk')
google_vision_api = "https://vision.googleapis.com/v1/images:annotate"


def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def apply_brightness_contrast(input_img, brightness, contrast, gamma):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        
    gray = cv2.cvtColor(buf, cv2.COLOR_BGR2GRAY)
    
    return adjust_gamma(gray,gamma)

def adjustColor(path):
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("grayscaled.jpg", img_gray)
    
    average = np.average(img_gray)
    # print (average)
    
    if (average > 199):
        img_temp = cv2.imread(path)
        img = apply_brightness_contrast(img_temp, -100, 130, 2)
        return (img)
        #cv2.imwrite("adjusted.jpg", img)
        
    elif (average < 200 and average > 150):
        img_temp = cv2.imread(path)
        img = apply_brightness_contrast(img_temp, -50, 110, 2)
        return (img)
        #cv2.imwrite("adjusted.jpg", img)
        
    elif (average < 151 and average > 110):
        img_temp = cv2.imread(path)
        img = apply_brightness_contrast(img_temp, -20, 90, 4)
        return (img)
        #cv2.imwrite("adjusted.jpg", img)
    elif (average < 111):
        img_temp = cv2.imread(path)
        img = apply_brightness_contrast(img_temp, 50, 80, 2)
        return (img)
        #cv2.imwrite("adjusted.jpg",img)

def scaleImg(img):
    height, width = img.shape[:2]
    new_width = 2918
    new_height = 3929
    # get scaling factor
    scaling_factor_x = new_width / float(width)
    scaling_factor_y = new_height / float(height)

    # resize image
    img = cv2.resize(img, None, fx=scaling_factor_x, fy=scaling_factor_y, interpolation=cv2.INTER_AREA)

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

    width = template_pdf.pages[0].MediaBox[2]
    height = template_pdf.pages[0].MediaBox[3]

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
                        index = (annotation[BOX_KEY].index(point))
                        if (index == 0 or index == 2):
                            box.append(float(point)/float(width))
                            # box.append((float(point) - 40)/((float(width))- (float(82/3000) * float(width))))
                        else:
                            box.append(float(point)/float(height))
                            # box.append((float(point) - 31)/((float(height))- (float(71/4000) * float(height))))
                            # box.append((float(point) - 31)/((float(height) - (71/4000 * float(height)))))

                    bounding_boxes[i][annotation[PARENT_KEY][ANNOT_FIELD_KEY][1:-1]].append({'box': box, 'type': type})

                if annotation[ANNOT_FIELD_KEY]:
                    if annotation[ANNOT_FIELD_KEY][1:-1] not in bounding_boxes[i].keys():
                        bounding_boxes[i][annotation[ANNOT_FIELD_KEY][1:-1]] = []

                    box = []

                    type = 'checkbox' if annotation[FIELD_TYPE_KEY] == CHECKBOX_KEY else 'text'
    
                    for point in annotation[BOX_KEY]:
                        index = (annotation[BOX_KEY].index(point))
                        if (index == 0 or index == 2):
                            # box.append((float(point) - 40)/((float(width))- (float(82/3000) * float(width))))
                            box.append(float(point)/float(width))
                        else:
                            box.append(float(point)/float(height))
                            # box.append((float(point) - 31)/((float(height))- (float(71/4000) * float(height))))
                            # box.append((float(point) - 31)/((float(height) - (71/4000 * float(height)))))
                        # print (point)
                    bounding_boxes[i][annotation[ANNOT_FIELD_KEY][1:-1]].append({'box': box, 'type': type})
        # print (bounding_boxes)
    
    for i in range(len(img_paths)):
        form = cv2.imread(img_paths[i])
        width = len(form[0])
        height = len(form)
        # print (width)
        # print(height)
        for key in bounding_boxes[i].keys():
            for j, box in enumerate(bounding_boxes[i][key]):
                if key == 'Reset':
                    continue

                left_x = (box['box'][0] * (width) - 40) * (3000/2918) - 40 #+ (3000/612*40)
                right_x = (box['box'][2]  * (width) - 40) * (3000/2918) + 20 #+ (3000/612*40)
                top_y = ((height - box['box'][1] * (height)) - 75) * (4000/3929) + 20 #- (4000/792*40/2)
                bot_y = ((height - box['box'][3]  * (height)) - 75) * (4000/3929) - 10 #- (4000/792*40/2)

                bounding_boxes[i][key][j]['box'] = [left_x, bot_y, right_x, top_y]
        print (height)
        print (width)
    # print(bounding_boxes)
    form = scaleImg(cv2.imread(img_paths[0]))

    for key in bounding_boxes[1]:
        for box in bounding_boxes[1][key]:
            if key == 'Reset':
                continue
            cv2.rectangle(form, (int(box['box'][0]), int(box['box'][3] )),
                        (int(box['box'][2] ), int(box['box'][1] )), (0, 0, 0), 5)

    cv2.imwrite("boundingBoxes.jpg", form)
    return bounding_boxes


# calls google ocr to get the handwritten text from a field in the form
# inputs: cv2 image of the field
# outputs: {'text': text in the field, 'confidence': confidence score}
async def getRegionText(img):
    try:
        retval, buffer = cv2.imencode('.jpg', img)
    except:
        return {'text': '', 'confidence': -1}

    image64 = base64.b64encode(buffer)

    params = {'key': google_vision_api_key}
    body = {"requests":[{"imageContext": {'languageHints' : ['en-t-i0-handwrit']}, "image":{"content": str(image64)[2:-1]},"features":[{"type":"DOCUMENT_TEXT_DETECTION","maxResults":1}]}]}

    async with ClientSession() as session:
        async with session.post(google_vision_api, params=params, json=body) as resp:
            resp = await resp.json()
            try:
                text = resp['responses'][0]['fullTextAnnotation']['text']
            except:
                text = ''

            try:
                confidence = resp['responses'][0]['fullTextAnnotation']['pages'][0]['blocks'][0]['confidence']
            except:
                confidence = -1

            return {'text': ' '.join(text.split('\n')), 'confidence': confidence}


# gets the text for all regions in a form page
# inputs: form: cv2 image of the form page
# inputs: regions: dict of bounding boxes for each field on the page
# inputs: auto_mode (optional): uses the automatic bounding boxes instead of the manual ones
# outputs: [keys of fields, values in fields, confidence of field]
async def callGoogle(form, regions, auto_mode=False):
    width = len(form[0])
    height = len(form)
    keys = []
    tasks = []
    if auto_mode:
        for key in regions.keys():
            for box in regions[key]:
                if box['type'] == 'text':
                    crop_img = form[int(box['box'][1]):int(box['box'][3]),
                               int(box['box'][0]):int(box['box'][2])]
                    res = asyncio.ensure_future(getRegionText(crop_img))
                    keys.append(key)
                    tasks.append(res)
    else:
        for key in regions.keys():
            for box in regions[key]:
                if box['type'] == 'text':
                    crop_img = form[int(box['box'][1]):int(box['box'][3]),
                               int(box['box'][0]):int(box['box'][2])]
                    res = asyncio.ensure_future(getRegionText(crop_img))
                    keys.append(key)
                    tasks.append(res)

    results = await asyncio.gather(*tasks)
    # print(tasks)
    # print(results)
    printing = json.dumps(results, indent=4)
    print (printing)
    return [keys, results]


async def test():
    dirpath = os.getcwd()
    fieldBounds = getPdfBoxes('OAF.pdf', ['oafpbm/' + i for i in os.listdir( dirpath + '/oafpbm/') if i[len(i)-4:] == '.jpg'], skip_corners=True)
    form = adjustColor('./oafpbm/1.jpg')
    form = scaleImg(form)
    regions = fieldBounds[0]
    count  = 0
    for key in regions.keys():
        for box in regions[key]:
            # print(str(box['box']))
            if box['type'] == 'text':
                crop_img = form[int(box['box'][1]):int(box['box'][3]),
                            int(box['box'][0]):int(box['box'][2])]
                cv2.imwrite('./Cropped/' + str(count) + '.jpg', crop_img)
                count+= 1

    img1 = adjustColor('./oafpbm/1.jpg')
    img1 = scaleImg(img1)
    future1 = await callGoogle(img1, fieldBounds[0])
    # print(future1)

loop = asyncio.new_event_loop()
loop.run_until_complete(test())
