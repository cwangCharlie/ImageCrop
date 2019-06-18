import cv2
import numpy as np
import argparse
import imutils
import pdfrw
import os

def scaleImg(img):
    height, width = img.shape[:2]
    new_width = 1275
    new_height = 1675
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
#     width = template_pdf[SIZE_KEY]
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
                        else:
                            box.append(float(point)/float(height))
#                     print (bounding_boxes)
                    bounding_boxes[i][annotation[PARENT_KEY][ANNOT_FIELD_KEY][1:-1]].append({'box': box, 'type': type})

                if annotation[ANNOT_FIELD_KEY]:
                    if annotation[ANNOT_FIELD_KEY][1:-1] not in bounding_boxes[i].keys():
                        bounding_boxes[i][annotation[ANNOT_FIELD_KEY][1:-1]] = []

                    box = []

                    type = 'checkbox' if annotation[FIELD_TYPE_KEY] == CHECKBOX_KEY else 'text'
    
                    for point in annotation[BOX_KEY]:
                        index = (annotation[BOX_KEY].index(point))
                        if (index == 0 or index == 2):
                            box.append(float(point)/float(width))
                        else:
                            box.append(float(point)/float(height))

                    bounding_boxes[i][annotation[ANNOT_FIELD_KEY][1:-1]].append({'box': box, 'type': type})
#     print (bounding_boxes)
    
    for i in range(len(img_paths)):
        form = cv2.imread(img_paths[i])
        width = len(form[0])
        height = len(form)
        print (width)
        print(height)
        for key in bounding_boxes[i].keys():
            for j, box in enumerate(bounding_boxes[i][key]):
                if key == 'Reset':
                    continue

                left_x = box['box'][0] * (width)
                right_x = box['box'][2]  * (width)
                top_y = height - box['box'][1] * (height)
                bot_y = height - box['box'][3]  * (height) 

                bounding_boxes[i][key][j]['box'] = [left_x, bot_y, right_x, top_y]
    print(bounding_boxes)
    form = scaleImg(cv2.imread(img_paths[1]))

    for key in bounding_boxes[0]:
        for box in bounding_boxes[0][key]:
            if key == 'Reset':
                continue
            cv2.rectangle(form, (int(box['box'][0]), int(box['box'][3] )),
                        (int(box['box'][2] ), int(box['box'][1] )), (100, 166, 189), 3)

    cv2.imwrite("boundingBoxes.jpg", form)
    return bounding_boxes

dirpath = os.getcwd()
fieldBounds = getPdfBoxes('OAF1.pdf', ['oafpbm/' + i for i in os.listdir(dirpath + '/oafpbm/') if i[len(i)-4:] == '.jpg'], skip_corners=True)
