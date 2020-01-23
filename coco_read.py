import random
from pycocotools.coco import COCO
import cv2
import json
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

MIN_KEYPOINTS = 10
MIN_HEIGHT = 150
SSIZE = 384  # size of cropped square

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# specify dataset path
dataDir='/Users/fabianschramm/Documents/coco'
dataType='train2017'
# dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
coco_kps=COCO(annFile)

# get all images containing given person instance (only them labeled with keypoints)
catIds = coco.getCatIds(catNms=['person'])  # 1
imgIds = coco.getImgIds(catIds=catIds )  # for train2017 len(imgIds) = 64,115 , val2017 len(imgIds) = 2693
print('in total {} images to process'.format(len(imgIds)))
# random select
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

s = 0  # TODO debug variable to break for loop
xy_crop = {}
for imgId in imgIds:
    if s > 10e100:
        break
    s = s + 1
    # load image
    img = coco.loadImgs(imgId)[0]
    # I = io.imread('%s/images/%s/%s' % (dataDir, dataType, img['file_name']))
    I2 = cv2.imread('%s/images/%s/%s' % (dataDir, dataType, img['file_name']))
    img_name, _ = img['file_name'].split('.')

    annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco_kps.loadAnns(annIds)
    for ann in anns:
        if ann['num_keypoints'] > MIN_KEYPOINTS and ann['bbox'][3] > MIN_HEIGHT:  # get just well annotated and not to small persons
            x, y, w, h = ann['bbox']
            x = x - (w*0.2)
            y = y - (h*0.2)
            w *= 1.4
            h *= 1.4

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            # crops image according to person plus margin
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + w > img['width']:
                w = img['width']
            else:
                w = x + w
            if y + h > img['height']:
                h = img['height']
            else:
                h = y + h

            # crop square of size SSIZE NOT WORKING
            # cx = x + w/2
            # cy = y + h/2
            # x = cx - SSIZE/2
            # y = cy - SSIZE/2
            #
            # x = int(x)
            # y = int(y)
            # w = int(w)
            # h = int(h)
            #
            # if x < 0:
            #     x = 0
            # if y < 0:
            #     y = 0
            #
            # if x + SSIZE > img['width']:
            #     w = (x+SSIZE) + int((img['width']) - (x+SSIZE))
            # else:
            #     w = SSIZE
            # if y + SSIZE > img['height']:
            #     h = (y+SSIZE) + int((img['height']) - (y+SSIZE))
            # else:
            #     h = SSIZE



            crop_img2 = I2[y:h, x:w]
            # cv2.imshow("cropped", crop_img2)
            # cv2.waitKey(0)
            im_path = '{}/images_cropped/{}/{}_annid{}.jpg'.format(dataDir, dataType, img_name, ann['id'])
            cv2.imwrite(im_path, crop_img2)

            # safe x and y offset from image crop to later adjust keypoint locations
            xy_crop[ann['id']] = (x, y)

    print('{} done so far'.format(s))

# dict to access how image was cropped in order to adjust keypoints and bbox
with open('{}/images_cropped/{}/xy_offset'.format(dataDir, dataType), 'w') as fp:
    json.dump(xy_crop, fp)

    # save picture with annotations
    # plt.imshow(I); plt.axis('off')
    # ax = plt.gca()
    # annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)  # catIds = 1 for just persons, annIds contains annotations for all persons
    # anns = coco_kps.loadAnns(annIds)
    # coco_kps.showAnns(anns)
    # plt.savefig('test{}.png'.format(img['id']))

