import math
import numpy as np
import chainer
from chainer.backends.cuda import get_array_module
from chainer import reporter
import chainer.functions as F
import chainer.links as L
from chainer import initializers
if chainer.backends.cuda.available:
    import cupy as xp

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

major, _, _ = chainer.__version__.split(".")
MAJOR = int(major)
if MAJOR >= 5:
    from chainer import static_graph
else:
    def static_graph(func):
        """
        dummy decorator to keep compatibility between Chainer v5 and v4
        """

        def wrap(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        return wrap

EPSILON = 1e-6


def area(bbox):
    _, _, w, h = bbox
    return w * h


def intersection(bbox0, bbox1):
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1

    w = F.relu(F.minimum(x0 + w0 / 2, x1 + w1 / 2) - F.maximum(x0 - w0 / 2, x1 - w1 / 2))
    h = F.relu(F.minimum(y0 + h0 / 2, y1 + h1 / 2) - F.maximum(y0 - h0 / 2, y1 - h1 / 2))

    return w * h


def iou(bbox0, bbox1):
    area0 = area(bbox0)
    area1 = area(bbox1)
    intersect = intersection(bbox0, bbox1)

    return intersect / (area0 + area1 - intersect + EPSILON)


def get_network(model, **kwargs):
    if model == 'mv2':
        from network_mobilenetv2 import MobilenetV2
        return MobilenetV2(**kwargs)
    elif model == 'resnet50':
        from network_resnet import ResNet50
        return ResNet50(**kwargs)
    elif model == 'resnet18':
        from network_resnet import ResNet
        return ResNet(n_layers=18)
    elif model == 'resnet34':
        from network_resnet import ResNet
        return ResNet(n_layers=34)
    else:
        raise Exception('Invalid model name')


class PoseProposalNet(chainer.link.Chain):

    def __init__(self,
                 model_name,
                 insize,
                 keypoint_names,
                 edges,
                 local_grid_size,
                 parts_scale,
                 instance_scale,
                 width_multiplier=1.0,
                 lambda_resp=0.25,
                 lambda_iou=1.0,
                 lambda_coor=5.0,
                 lambda_size=5.0,
                 lambda_limb=0.5,
                 dtype=np.float32):
        super(PoseProposalNet, self).__init__()
        self.model_name = model_name
        self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.local_grid_size = local_grid_size
        self.dtype = dtype
        self.lambda_resp = lambda_resp
        self.lambda_iou = lambda_iou
        self.lambda_coor = lambda_coor
        self.lambda_size = lambda_size
        self.lambda_limb = lambda_limb
        self.parts_scale = np.array(parts_scale)
        self.instance_scale = np.array(instance_scale)
        with self.init_scope():
            self.feature_layer = get_network(model_name, dtype=dtype, width_multiplier=width_multiplier)
            ksize = self.feature_layer.last_ksize
            self.lastconv = L.Convolution2D(None,
                                            6 * len(self.keypoint_names) +
                                            self.local_grid_size[0] * self.local_grid_size[1] * len(self.edges),
                                            ksize=ksize, stride=1, pad=ksize // 2,
                                            initialW=initializers.HeNormal(1 / np.sqrt(2), dtype))

        self.outsize = self.get_outsize()
        inW, inH = self.insize
        outW, outH = self.outsize
        self.gridsize = (int(inW / outW), int(inH / outH))
        num_param = self.feature_layer.count_params()
        num_param2 = self.lastconv.count_params()
        logger.info('number of mobilnet parameters: {}'.format(num_param))
        logger.info('number of last layer parameters: {}'.format(num_param2))
        logger.info('number of total parameters: {}'.format(num_param+num_param2))


    def get_outsize(self):
        inp = np.zeros((2, 3, self.insize[1], self.insize[0]), dtype=np.float32)
        out = self.forward(inp)
        _, _, h, w = out.shape
        return w, h

    def restore_xy(self, x, y):
        xp = get_array_module(x)
        gridW, gridH = self.gridsize
        outW, outH = self.outsize
        X, Y = xp.meshgrid(xp.arange(outW, dtype=xp.float32), xp.arange(outH, dtype=xp.float32))
        return (x + X) * gridW, (y + Y) * gridH

    def restore_size(self, w, h):
        inW, inH = self.insize
        return inW * w, inH * h

    def encode(self, in_data):
        image = in_data['image']
        keypoints = in_data['keypoints']
        bbox = in_data['bbox']
        is_labeled = in_data['is_labeled']
        dataset_type = in_data['dataset_type']
        inW, inH = self.insize
        outW, outH = self.outsize
        gridW, gridH = self.gridsize
        K = len(self.keypoint_names)

        # responsability delta
        delta = np.zeros((K, outH, outW), dtype=np.float32)
        tx = np.zeros((K, outH, outW), dtype=np.float32)
        ty = np.zeros((K, outH, outW), dtype=np.float32)
        tw = np.zeros((K, outH, outW), dtype=np.float32)
        th = np.zeros((K, outH, outW), dtype=np.float32)

        # predicted edges
        te = np.zeros((
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW), dtype=np.float32)

        # Set delta^i_k
        # loop over several people
        for (x, y, w, h), points, labeled in zip(bbox, keypoints, is_labeled):
            if dataset_type == 'mpii':
                partsW, partsH = self.parts_scale * math.sqrt(w * w + h * h) # half head diagonal
                instanceW, instanceH = self.instance_scale * math.sqrt(w * w + h * h) # double head diagonal
            elif dataset_type == 'coco':
                partsW, partsH = self.parts_scale * math.sqrt(w * w + h * h)
                instanceW, instanceH = w, h
            else:
                raise ValueError("must be 'mpii' or 'coco' but actual {}".format(dataset_type))
            # get center of head bounding box
            cy = y + h / 2
            cx = x + w / 2
            points = [[cy, cx]] + list(points) # add center of head to list of keypoints
            labeled = [True] + list(labeled)
            for k, (yx, l) in enumerate(zip(points, labeled)):
                if not l:
                    continue
                cy = yx[0] / gridH
                cx = yx[1] / gridW
                ix, iy = int(cx), int(cy) # index of center point
                sizeW = instanceW if k == 0 else partsW # head
                sizeH = instanceH if k == 0 else partsH # rest of keypoints
                if 0 <= iy < outH and 0 <= ix < outW: # if key point in output grid
                    delta[k, iy, ix] = 1 # get the responsible cell and set 1
                    tx[k, iy, ix] = cx - ix # construct the bounding box around joints
                    ty[k, iy, ix] = cy - iy
                    tw[k, iy, ix] = sizeW / inW
                    th[k, iy, ix] = sizeH / inH

            for ei, (s, t) in enumerate(self.edges):
                if not labeled[s]:
                    continue
                if not labeled[t]:
                    continue
                src_yx = points[s]
                tar_yx = points[t]
                # get index of joint in grid
                # gridH, gridW is gridsize ratio between inW / outW and inH/outH (one step in x direction of output is this ratio step in input)
                iyx = (int(src_yx[0] / gridH), int(src_yx[1] / gridW))
                jyx = (int(tar_yx[0] / gridH) - iyx[0] + self.local_grid_size[1] // 2, # why this addition of local grid size //2
                       int(tar_yx[1] / gridW) - iyx[1] + self.local_grid_size[0] // 2)

                if iyx[0] < 0 or iyx[1] < 0 or iyx[0] >= outH or iyx[1] >= outW:
                    continue
                if jyx[0] < 0 or jyx[1] < 0 or jyx[0] >= self.local_grid_size[1] or jyx[1] >= self.local_grid_size[0]: # check if reachable
                    continue

                te[ei, jyx[0], jyx[1], iyx[0], iyx[1]] = 1

        # define max(delta^i_k1, delta^j_k2) which is used for loss_limb
        max_delta_ij = np.ones((len(self.edges),
                                outH, outW,
                                self.local_grid_size[1], self.local_grid_size[0]), dtype=np.float32)
        or_delta = np.zeros((len(self.edges), outH, outW), dtype=np.float32)
        for ei, (s, t) in enumerate(self.edges):
            or_delta[ei] = np.minimum(delta[s] + delta[t], 1)
        mask = F.max_pooling_2d(np.expand_dims(or_delta, axis=0),
                                ksize=(self.local_grid_size[1], self.local_grid_size[0]),
                                stride=1,
                                pad=(self.local_grid_size[1] // 2, self.local_grid_size[0] // 2))
        mask = np.squeeze(mask.array, axis=0)
        for index, _ in np.ndenumerate(mask):
            max_delta_ij[index] *= mask[index]
        max_delta_ij = max_delta_ij.transpose(0, 3, 4, 1, 2)

        # preprocess image
        image = self.feature_layer.prepare(image)

        return image, delta, max_delta_ij, tx, ty, tw, th, te

    def _forward(self, x):
        h = F.cast(x, self.dtype)
        h = self.feature_layer(h)
        h = self.feature_layer.last_activation(self.lastconv(h))
        # print(cupy.get_default_memory_pool().used_bytes())
        return h

    @static_graph
    def static_forward(self, x):
        return self._forward(x)

    def forward(self, x):
        """
        This provides an interface of forwarding.
        ChainerV5 has a feature Static Subgraph Optimizations to increase training speed.
        But for some reason, our model does not decrease loss value at all.
        We do not trust it for now on training. On the other hand, by decorating `static_graph`
        at forward function, it increases speed of inference very well.
        Also note that if we use ideep option, the output result between 
        `static_forward` and `_forward` will be different.
        """
        if chainer.config.train:
            return self._forward(x)
        else:
            if MAJOR >= 5 and chainer.backends.cuda.available:
                return self.static_forward(x)
            else:
                return self._forward(x)

    def __call__(self, image, delta, max_delta_ij, tx, ty, tw, th, te):
        K = len(self.keypoint_names)
        B, _, _, _ = image.shape
        outW, outH = self.outsize
        # print(image.nbytes)
        feature_map = self.forward(image)
        resp = feature_map[:, 0 * K:1 * K, :, :]
        conf = feature_map[:, 1 * K:2 * K, :, :]
        x = feature_map[:, 2 * K:3 * K, :, :]
        y = feature_map[:, 3 * K:4 * K, :, :]
        w = feature_map[:, 4 * K:5 * K, :, :]
        h = feature_map[:, 5 * K:6 * K, :, :]
        # edges e
        e = feature_map[:, 6 * K:, :, :].reshape((
            B,
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW
        ))

        (rx, ry), (rw, rh) = self.restore_xy(x, y), self.restore_size(w, h)
        (rtx, rty), (rtw, rth) = self.restore_xy(tx, ty), self.restore_size(tw, th)
        ious = iou((rx, ry, rw, rh), (rtx, rty, rtw, rth))

        # add weight where can't find keypoint
        xp = get_array_module(max_delta_ij)
        zero_place = xp.zeros(max_delta_ij.shape).astype(self.dtype)
        zero_place[max_delta_ij < 0.5] = 0.0005
        weight_ij = xp.minimum(max_delta_ij + zero_place, 1.0)

        xp = get_array_module(delta)
        # add weight where can't find keypoint
        zero_place = xp.zeros(delta.shape).astype(self.dtype)
        zero_place[delta < 0.5] = 0.0005
        weight = xp.minimum(delta + zero_place, 1.0)

        half = xp.zeros(delta.shape).astype(self.dtype)
        half[delta < 0.5] = 0.5

        loss_resp = F.sum(F.square(resp - delta), axis=tuple(range(1, resp.ndim)))
        loss_iou = F.sum(delta * F.square(conf - ious), axis=tuple(range(1, conf.ndim)))
        loss_coor = F.sum(weight * (F.square(x - tx - half) + F.square(y - ty - half)), axis=tuple(range(1, x.ndim)))
        loss_size = F.sum(weight * (F.square(F.sqrt(w + EPSILON) - F.sqrt(tw + EPSILON)) +
                                    F.square(F.sqrt(h + EPSILON) - F.sqrt(th + EPSILON))),
                          axis=tuple(range(1, w.ndim)))
        loss_limb = F.sum(weight_ij * F.square(e - te), axis=tuple(range(1, e.ndim)))

        loss_resp = F.mean(loss_resp)
        loss_iou = F.mean(loss_iou)
        loss_coor = F.mean(loss_coor)
        loss_size = F.mean(loss_size)
        loss_limb = F.mean(loss_limb)

        loss = self.lambda_resp * loss_resp + \
            self.lambda_iou * loss_iou + \
            self.lambda_coor * loss_coor + \
            self.lambda_size * loss_size + \
            self.lambda_limb * loss_limb

        reporter.report({
            'loss': loss,
            'loss_resp': loss_resp,
            'loss_iou': loss_iou,
            'loss_coor': loss_coor,
            'loss_size': loss_size,
            'loss_limb': loss_limb
        }, self)

        return loss

    def predict(self, image):
        K = len(self.keypoint_names)
        B, _, _, _ = image.shape
        outW, outH = self.outsize

        with chainer.using_config('train', False),\
                chainer.function.no_backprop_mode():
            feature_map = self.forward(image)

        resp = feature_map[:, 0 * K:1 * K, :, :]
        conf = feature_map[:, 1 * K:2 * K, :, :]
        x = feature_map[:, 2 * K:3 * K, :, :]
        y = feature_map[:, 3 * K:4 * K, :, :]
        w = feature_map[:, 4 * K:5 * K, :, :]
        h = feature_map[:, 5 * K:6 * K, :, :]
        e = feature_map[:, 6 * K:, :, :].reshape((
            B,
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW
        ))

        return resp, conf, x, y, w, h, e

    #### TODO right inference for image set

    def predict_video(self, image_set):
        image_set = xp.asarray(image_set, dtype=xp.float32)

        image_set = xp.moveaxis(image_set, 3, 1)
        image_set = xp.ascontiguousarray(image_set, dtype=xp.float32)
        K = len(self.keypoint_names)
        B, _, _, _ = image_set.shape
        outW, outH = self.outsize

        with chainer.using_config('train', False),\
                chainer.function.no_backprop_mode():
            feature_map = self.forward(image_set)

        resp = feature_map[:, 0 * K:1 * K, :, :]
        conf = feature_map[:, 1 * K:2 * K, :, :]
        x = feature_map[:, 2 * K:3 * K, :, :]
        y = feature_map[:, 3 * K:4 * K, :, :]
        w = feature_map[:, 4 * K:5 * K, :, :]
        h = feature_map[:, 5 * K:6 * K, :, :]
        e = feature_map[:, 6 * K:, :, :].reshape((
            B,
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW
        ))
        resp = chainer.backends.cuda.to_cpu(resp.array)
        conf = chainer.backends.cuda.to_cpu(conf.array)
        w = chainer.backends.cuda.to_cpu(w.array)
        h = chainer.backends.cuda.to_cpu(h.array)
        x = chainer.backends.cuda.to_cpu(x.array)
        y = chainer.backends.cuda.to_cpu(y.array)
        e = chainer.backends.cuda.to_cpu(e.array)

        return resp, conf, x, y, w, h, e
