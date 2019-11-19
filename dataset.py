import os

from chainer.dataset import DatasetMixin
from chainercv import utils
import chainercv.transforms as transforms
import numpy as np

from augment import random_rotate, random_flip, random_crop
from augment import scale_fit_short, resize, resize_to_scale
from augment import augment_image
from utils import show_image1


class KeypointDataset2D(DatasetMixin):

    def __init__(self,
                 dataset_type,
                 insize,
                 keypoint_names,
                 edges,
                 flip_indices,
                 keypoints,
                 bbox,
                 is_visible,
                 is_labeled,
                 scale,
                 position,
                 image_paths,
                 image_root='.',
                 use_cache=False,
                 do_augmentation=False
                 ):
        self.dataset_type = dataset_type
        self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.flip_indices = flip_indices
        self.keypoints = keypoints  # [array of y,x]
        self.bbox = bbox  # [x,y,w,h]
        self.is_visible = is_visible
        self.is_labeled = is_labeled
        self.scale = scale
        self.position = position
        self.image_paths = image_paths
        self.image_root = image_root
        self.do_augmentation = do_augmentation
        self.use_cache = use_cache
        self.cached_samples = [None] * len(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def transform(self, image, keypoints, bbox, is_labeled, is_visible, dataset_type, scale):
        transform_param = {}

        # Color augmentation
        # image, param = augment_image(image, dataset_type)
        # transform_param['augment_image'] = param
        transform_param['augment_image'] = "0"

        # Random rotate
        image, keypoints, bbox, param = random_rotate(image, keypoints, bbox)
        transform_param['random_rotate'] = param

        # Random flip
        image, keypoints, bbox, is_labeled, is_visible, param = random_flip(image, keypoints, bbox, is_labeled, is_visible, self.flip_indices)
        transform_param['random_flip'] = param

        # Random crop
        # image, keypoints, bbox, param = random_crop(image, keypoints, bbox, is_labeled, dataset_type)
        # transform_param['random_crop'] = param
        transform_param['random_crop'] = "0"

        # scale to approx 200 px
        # image, keypoints, bbox = resize_to_scale(image, keypoints, bbox, scale=scale, insize=self.insize)
        return image, keypoints, bbox, is_labeled, is_visible, transform_param

    def get_example(self, i):
        w, h = self.insize

        if self.use_cache and self.cached_samples[i] is not None:
            image, keypoints, bbox, is_labeled, is_visible = self.cached_samples[i]
        else:
            path = os.path.join(self.image_root, self.image_paths[i])
            image = utils.read_image(path, dtype=np.float32, color=True)
            keypoints = self.keypoints[i]
            bbox = self.bbox[i]
            is_labeled = self.is_labeled[i]
            is_visible = self.is_visible[i]
            scale = self.scale[i]
            position = self.position[i]

            if self.use_cache:
                image, keypoints, bbox = resize(image, keypoints, bbox, (h, w))
                self.cached_samples[i] = image, keypoints, bbox, is_labeled, is_visible

        image = image.copy()
        keypoints = keypoints.copy()
        bbox = bbox.copy()
        is_labeled = is_labeled.copy()
        is_visible = is_visible.copy()
        scale = scale.copy()
        # position = position.copy()

        transform_param = {}
        try:
            # put it here for testing with provided MPII dataset - otherwise scale of persons is
            # to different and can't detect limbs with grid 3x3 -> huge error for limbs
            # usually should not be necessary when testing with proper video dataset
            image, keypoints, bbox = resize_to_scale(image, keypoints, bbox, scale=scale, insize=self.insize)

            if self.do_augmentation:
                # image, keypoints, bbox = scale_fit_short(image, keypoints, bbox, length=int(min(h, w) * 1.25))
                # TODO activate this line here and uncomment above resize_to_scale when testing with real data
                # image, keypoints, bbox = resize_to_scale(image, keypoints, bbox, scale=scale, insize=self.insize)
                # utils.write_image(image, os.path.join('/home/fabian/Desktop/test/', self.image_paths[i]))

                # adapted transforms, random crop destroys scaling to normalized
                image, keypoints, bbox, is_labeled, is_visible, transform_param = self.transform(
                    image, keypoints, bbox, is_labeled, is_visible, self.dataset_type, scale)
                # utils.write_image(image, os.path.join('/home/fabian/Desktop/test4/', self.image_paths[i]))

                # should be one intend level up when tested with real dataset and resize_to_scale is deactivated
                image, keypoints, bbox = resize(image, keypoints, bbox, (h, w))

            transform_param['do_augmentation'] = self.do_augmentation

        except Exception as e:
            raise Exception("something wrong...transform_param = {}".format(transform_param))

        return {
            'path': self.image_paths[i],
            'keypoint_names': self.keypoint_names,
            'edges': self.edges,
            'image': image,
            'keypoints': keypoints,
            'bbox': bbox,
            'is_labeled': is_labeled,
            'is_visible': is_visible,
            'dataset_type': self.dataset_type,
            'transform_param': transform_param
        }
