import random
from pycocotools.coco import COCO
import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# specify dataset path
dataDir='/Users/fabianschramm/Documents/coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))

# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given person instance (only them labeled with keypoints)
catIds = coco.getCatIds(catNms=['person'])  # 1
imgIds = coco.getImgIds(catIds=catIds )  # for train2017 len(imgIds) = 64,115 , val2017 len(imgIds) = 2693
#imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# load and display image
I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
# I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
# plt.show()
plt.savefig('test.png')


# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
coco_kps=COCO(annFile)

# load and display keypoints annotations
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)  # catIds = 1 for just persons, annIds contains annotations for all persons
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
plt.savefig('test2.png')