import argparse
import configparser
import os
import queue
import threading
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import chainer
import cv2
import numpy as np
from PIL import Image

from predict import get_feature, get_humans_by_feature, draw_humans, create_model
from utils import parse_size
from multiprocessing import Process, Event

QUEUE_SIZE = 0

"""
Bonus script
If you have good USB camera which gets image as well as 60 FPS,
this script will be helpful for realtime inference
"""


class Capture(Process):

    def __init__(self, cap):
        super(Capture, self).__init__()
        self.cap = cap
        self.stop_event = Event()
        self.queue1 = queue.Queue(QUEUE_SIZE)
        self.queue2 = queue.Queue(QUEUE_SIZE)
        self.name = 'Capture'
        self.count = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                ret_val, image = self.cap.read()
                self.count += 1
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # move resizing to predictor
                # image = cv2.resize(image, self.insize)
                self.queue1.put((image, self.count), timeout=100)
                self.queue2.put((image, self.count), timeout=100)
                #print("read queue: ", self.queue1.qsize())
            except queue.Full:
                pass
            except cv2.error:
                print("cv2 error")
                pass

    def get(self, i):
        if i == 1:
            return self.queue1.get(timeout=100)
        else:
            return self.queue2.get(timeout=100)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()


class Predictor(Process):

    def __init__(self, model, cap):
        super(Predictor, self).__init__()
        self.cap = cap
        self.model = model
        self.stop_event = Event()
        self.queue = queue.Queue(QUEUE_SIZE)
        self.name = 'Predictor '+str(model.insize[0])+'x'+str(model.insize[1])
        self.insize = model.insize

    # def run(self):
    #     while not self.stop_event.is_set():
    #         try:
    #             image, count = self.cap.get()
    #             # print(count)
    #             image = cv2.resize(image, self.insize)
    #             with chainer.using_config('autotune', True), \
    #                     chainer.using_config('use_ideep', 'auto'):
    #                 feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
    #             self.queue.put((image, feature_map), timeout=1)
    #             print("pred queue: ", self.queue.qsize())
    #         except queue.Full:
    #             pass
    #         except queue.Empty:
    #             pass

    def get(self):
        return self.queue.get(timeout=100)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()


class Predictor1(Predictor):
    def run(self):
        while not self.stop_event.is_set():
            try:
                image, count = self.cap.get(1)
                print('pred1 getting from cap:'+str(count)+'\n')
                image = cv2.resize(image, self.insize)
                with chainer.using_config('autotune', True), \
                     chainer.using_config('use_ideep', 'auto'):
                    feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
                self.queue.put((image, feature_map), timeout=100)
                print("pred1 queue: ", self.queue.qsize())
            except queue.Full:
                pass
            except queue.Empty:
                pass


class Predictor2(Predictor):
    def run(self):
        while not self.stop_event.is_set():
            try:
                image, count = self.cap.get(2)
                print('pred2 getting from cap:'+str(count)+'\n')
                image = cv2.resize(image, self.insize)
                with chainer.using_config('autotune', True), \
                     chainer.using_config('use_ideep', 'auto'):
                    feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
                self.queue.put((image, feature_map), timeout=100)
                print("pred2 queue: ", self.queue.qsize())
            except queue.Full:
                pass
            except queue.Empty:
                pass


def load_config(args):
    config1 = configparser.ConfigParser()
    config_path1 = os.path.join(args.model1, 'src', 'config.ini')
    logger.info(config_path1)
    config1.read(config_path1, 'UTF-8')

    config2 = configparser.ConfigParser()
    config_path2 = os.path.join(args.model2, 'src', 'config.ini')
    logger.info(config_path2)
    config2.read(config_path2, 'UTF-8')
    return config1, config2



def high_speed(args):
    # model1 is 1920x1080
    config1, config2 = load_config(args)
    dataset_type1 = config1.get('dataset', 'type')
    detection_thresh1 = config1.getfloat('predict', 'detection_thresh')
    min_num_keypoints1 = config1.getint('predict', 'min_num_keypoints')
    model1 = create_model(args.model1, config1)

    # model2 is 224x224
    dataset_type2 = config2.get('dataset', 'type')
    detection_thresh2 = config2.getfloat('predict', 'detection_thresh')
    min_num_keypoints2 = config2.getint('predict', 'min_num_keypoints')
    model2 = create_model(args.model2, config2)
    model2.to_gpu(2) # TODO GET DEVICE RIGHT

    if os.path.exists('mask.png'):
        mask = Image.open('mask.png')
        mask = mask.resize((200, 200))
    else:
        mask = None

    # cap = cv2.VideoCapture(0) # get input from usb camera
    cap = cv2.VideoCapture("/home/fabian/Documents/dataset/videos/test4.mp4")
    if cap.isOpened() is False:
        print('Error opening video stream or file')
        exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info('camera will capture {} FPS'.format(cap.get(cv2.CAP_PROP_FPS)))



    capture = Capture(cap)
    # predictor1 = Predictor1(model=model1, cap=capture)
    # predictor2 = Predictor2(model=model2, cap=capture)

    capture.start()
    # predictor1.start()
    # predictor2.start()

    while True:
        pass

    # fps_time = 0
    # degree = 0
    #
    # main_event = threading.Event()
    #
    # try:
    #     while not main_event.is_set() and cap.isOpened():
    #         degree += 5
    #         degree = degree % 360
    #         try:
    #             image, feature_map = predictor1.get()
    #             image2, feature_map2 = predictor2.get()
    #             humans = get_humans_by_feature(
    #                 model1,
    #                 feature_map,
    #                 detection_thresh1,
    #                 min_num_keypoints1
    #             )
    #         except queue.Empty:
    #             continue
    #         except Exception:
    #             break
    #         pilImg = Image.fromarray(image)
    #         pilImg = draw_humans(
    #             model1.keypoint_names,
    #             model1.edges,
    #             pilImg,
    #             humans,
    #             mask=mask.rotate(degree) if mask else None,
    #             visbbox=config1.getboolean('predict', 'visbbox'),
    #         )
    #         img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
    #         msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
    #         msg += ' ' + config1.get('model_param', 'model_name')
    #         cv2.putText(img_with_humans, 'FPS: %f' % (1.0 / (time.time() - fps_time)),
    #                     (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         img_with_humans = cv2.resize(img_with_humans, (int(1/3 * model1.insize[0]), int(1/3 * model1.insize[1])))
    #         cv2.imshow('Pose Proposal Network' + msg, img_with_humans)
    #         fps_time = time.time()
    #         # press Esc to exit
    #         if cv2.waitKey(1) == 27:
    #             main_event.set()
    # except Exception as e:
    #     print(e)
    # except KeyboardInterrupt:
    #     main_event.set()
    #
    # capture.stop()
    # predictor1.stop()
    # predictor2.stop()
    #
    # capture.join()
    # predictor1.join()
    # predictor2.join()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model1', metavar='M1', help='path/to/model1', type=str)
    parser.add_argument('model2', metavar='M2', help='path/to/model2', type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    high_speed(args)


if __name__ == '__main__':
    main()
