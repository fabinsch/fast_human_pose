import argparse
import configparser
import os
import queue
import threading
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import chainercv.transforms as transforms
import sys

import chainer
import cv2
import numpy as np
from PIL import Image

from predict import get_feature, get_humans_by_feature, draw_humans, create_model
from utils import parse_size
from multiprocessing import Process, Event, set_start_method, Pipe, Queue
import copy
if chainer.backends.cuda.available:
    import cupy as xp
else:
    import cupy as xp


QUEUE_SIZE = 0

# from queue import Queue


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
                self.queue1.put((image, self.count), timeout=100)
                self.queue2.put((image, self.count), timeout=100)
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
        self.stop_event.set()


class Predictor(Process):

    def __init__(self, modelargs, config, queue_in, queue_comm, pipe_end, detection_threshold, min_num_keypoints): #, cap):
        super(Predictor, self).__init__()
        # self.cap = cap
        self.modelargs = modelargs
        self.config = config
        self.stop_event = Event()
        self.queue = Queue(QUEUE_SIZE)
        self.queue_in = queue_in
        self.queue_comm = queue_comm
        self.pipe_end = pipe_end
        # print(queue_in)
        self.model = None
        self.insize = None
        self.detection_threshold = detection_threshold
        self.min_num_keypoints = min_num_keypoints
        self.count = 0
        self.inf_time = 0
        self.queue_get_time = 0
        logger.info('{} initializing and loading model'.format(self.name))
        # insize = (1920, 1080)
        # self.name = 'Predictor '+str(insize[0])+'x'+str(insize[1])
        # self.insize = insize

    def get(self):
        return self.queue.get(timeout=100)

    def stop(self):
        self.stop_event.set()
        self.pipe_end.close()
        logger.info('{} will stop'.format(self.name))
        logger.info("{} processed {} images in {}".format(self.name, self.queue.qsize(), self.insize))
        logger.info("getting from queue average: {}s".format(self.queue_get_time / (self.queue.qsize())))
        logger.info("inference time average: {}s".format(self.inf_time / (self.queue.qsize() - 1)))  # substract one for the first image passed trough network

    def cut_human(self, image, humans):
        # loop through humans to cut image
        # always return 20 image crops
        image_set = []
        if len(humans) > 0:
            for h in humans[0].values():
                # cut image according to detected person
                # is seems like dimension always have to be the same to convert to cupy
                # TODO smart way to get same size instead of resizing
                image_set.append(
                    cv2.resize(
                        image[int(h[0]): int(h[2]), int(h[1]): int(h[3]), :],
                        (224, 224)
                ))
                # image_set=xp.array(image[int(h[0]): int(h[2]), int(h[1]): int(h[3]), :])
            # image_set.append(self.random_crop(im=image, exp= 20 - len(humans[0].keys())))
        else:
            image_set = self.random_crop(im=image, exp=20)

        return image_set

    def random_crop(self, im, exp):
        image_set = []
        for i in range(exp):
            image, param = transforms.random_crop(
                np.moveaxis(im, 2,0),
                size=(224, 224),
                return_param=True
            )
            image_set.append(np.moveaxis(image, 0, 2))
        return image_set


class Predictor1(Predictor):
    def run(self):
        model = create_model(self.modelargs, self.config)
        logger.info('{} started at PID {} - 1920x1080 model loaded'.format(self.name, self.pid))
        if chainer.backends.cuda.available:
            model = model.to_gpu(0)
        self.model = model
        self.insize = (1920, 1080)
        self.pipe_end.send(True)  # model loaded sign
        count = 0
        run = False
        if self.pipe_end.recv():
            logger.info("start running 1920x1080")
            run = True

        while not self.stop_event.is_set():
            try:
                if run and not self.pipe_end.poll():
                    t_start = time.time()
                    # image, count = self.cap.get(1)
                    image, count = self.queue_in.get(timeout=1)
                    self.queue_get_time += time.time() - t_start
                    # logger.info('get img from queue took {} sec'.format(time.time()-t_start))

                    # print('pred1 getting from cap:'+str(count)+'\n')
                    image = cv2.resize(image, self.insize)
                    t_start = time.time()
                    with chainer.using_config('autotune', True), \
                         chainer.using_config('use_ideep', 'auto'):
                        feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
                    if not self.queue.empty():
                        self.inf_time += time.time() - t_start
                    # self.queue.put((image, feature_map), timeout=1)
                    self.queue.put((feature_map), timeout=1)
                    humans = get_humans_by_feature(model, feature_map, self.detection_threshold, self.min_num_keypoints)
                    if len(humans) > 0:
                        self.queue_comm.put(humans, timeout=1)
                    # cropped_image_set = self.cut_human(image, humans)
                    # logger.debug("pred1 queue {}: ".format(self.queue.qsize()))
                    if self.queue.qsize() == 1: self.pipe_end.send(2)  # sign that big model passed first forward path

                    ## BLOCK ON PORPUSE ##
                    block = False
                    while block:
                        time.sleep(2)
                        pass
                    ######
                else:
                    # logger.info("waiting for other model to load....")
                    if self.pipe_end.recv() == 'stop':
                        logger.info("STOP received via pipe")
                        self.stop()
                    if self.queue_in.qsize()==0 and self.queue.qsize()>0:
                        self.pipe_end.send('stop')
                        self.stop()
            except queue.Full:
                logger.info("queue full")
                pass
            except queue.Empty:
                logger.info("queue empty")
                if self.queue.qsize() > 0:
                    self.stop()
                else:
                    pass
            except cv2.error:
                logger.info("CV2 error")
                self.pipe_end.send('stop')
                time.sleep(1)
                self.stop()
            except KeyboardInterrupt:
                self.pipe_end.send('stop')
                self.stop()
            except:
                print("Unexpected error:", sys.exc_info()[0])
                self.pipe_end.send('stop')
                time.sleep(1)
                self.stop()
                raise


class Predictor2(Predictor):
    def run(self):
        model = create_model(self.modelargs, self.config)
        logger.info('{} started at PID {} - 224x224 model loaded'.format(self.name, self.pid))
        if chainer.backends.cuda.available:
            model = model.to_gpu(1)
        self.model = model
        self.insize = (224, 224)
        self.pipe_end.send(True)  # model loaded sign
        count = 0
        run = False
        if self.pipe_end.recv():
            logger.info("start running 224x224")
            run = True

        # run the first forward path to get model auto tune right
        # than wait for bigger model to complete first forward path
        image, count = self.queue_in.get(timeout=1)
        image = cv2.resize(image, self.insize)
        with chainer.using_config('autotune', True), \
             chainer.using_config('use_ideep', 'auto'):
            feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
        # self.queue.put((image, feature_map), timeout=1)
        self.queue.put((feature_map), timeout=1)

        if self.pipe_end.recv() == 2:
            pass

        while not self.stop_event.is_set():
            try:
                if run:
                    t_start = time.time()
                    # image, count = self.cap.get(2)
                    image, count = self.queue_in.get(timeout=1)
                    self.queue_get_time += time.time()-t_start
                    try:
                        humans = self.queue_comm.get(timeout=1)
                        # if humans: print(humans)
                        cropped_image_set = self.cut_human(image, humans)
                    except queue.Empty:
                        logger.info('humans queue empty')
                        cropped_image_set = self.random_crop(image, 20)

                    feature_map = self.model.predict_video(cropped_image_set)

                    # logger.info('get img from queue took {} sec'.format(time.time()-t_start))
                    # print('pred2 getting from cap:'+str(count)+'\n')
                    image = cv2.resize(image, self.insize)
                    # print(cropped_image_set)
                    t_start = time.time()
                    with chainer.using_config('autotune', True), \
                         chainer.using_config('use_ideep', 'auto'):
                        feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
                    if not self.queue.empty():
                        self.inf_time = time.time() - t_start
                    # self.queue.put((image, feature_map), timeout=1)
                    self.queue.put((feature_map), timeout=1) # maybe not needed to be a queue, just internal storage of process
                    #logger.debug("pred2 queue: {}".format(self.queue.qsize()))

                else:
                    logger.info("waiting for other model to load....")
                    if self.pipe_end.recv() == 'stop':
                        print("STOP received via pipe")
                    if self.queue_in.qsize() == 0 and self.queue.qsize() > 0:
                        self.pipe_end.send('stop')
                        self.stop()

            except queue.Full:
                logger.info("queue full")
                pass
            except queue.Empty:
                logger.info("queue empty")
                if self.queue.qsize() > 0:  # self.pipe_end.recv() == 'stop':
                    self.pipe_end.send('stop')
                    self.stop()
                else:
                    pass
            except cv2.error:
                logger.info("CV2 error")
                logger.info('{} exiting'.format(self.name))
                self.pipe_end.send('stop')
                time.sleep(1)
                self.stop()
            except KeyboardInterrupt:
                self.pipe_end.send('stop')
                self.stop()
            except:
                print("Unexpected error:", sys.exc_info()[0])
                self.pipe_end.send('stop')
                time.sleep(1)
                self.stop()
                raise


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

    # it seems not possible to create GPU related models outside the process, move inside
    # model1 = create_model(args.model1, config1)

    # model2 is 224x224
    dataset_type2 = config2.get('dataset', 'type')
    detection_thresh2 = config2.getfloat('predict', 'detection_thresh')
    min_num_keypoints2 = config2.getint('predict', 'min_num_keypoints')

    # cap = cv2.VideoCapture(0) # get input from usb camera
    # cap = cv2.VideoCapture('/home/mech-user/Documents/fabian/chainer-pose-proposal-net/work/video/test.mp4')
    cap = cv2.VideoCapture("/home/fabian/Documents/dataset/videos/test4.mp4")
    # cap = cv2.VideoCapture("/home/mech-user/Documents/fabian/data/videos/test4.mp4")

    if cap.isOpened() is False:
        print('Error opening video stream or file')
        exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info('camera will capture {} FPS'.format(cap.get(cv2.CAP_PROP_FPS)))

    # queue_main = queue.Queue(QUEUE_SIZE)
    queue_main = Queue(QUEUE_SIZE)
    queue_comm = Queue(QUEUE_SIZE)
    counter = 0

    # first read in the whole video stream and later process by parallel running networks
    # TODO also do this in parallel
    ret_val = True
    t_start = time.time()
    while ret_val:
        ret_val, image = cap.read()
        if ret_val:
            # image = cv2.resize(image, (224, 224))  # get timings with smaller image size in queue
            if counter < 40: queue_main.put((image,  counter))
            counter += 1
        # pass
    logger.info('loading video with {} frames took: {} seconds'.format(counter, time.time()-t_start))
    # queue_copy = copy.copy(queue_main)
    # instantiate the processes
    # capture = Capture(cap)
    fast_conn, reg_conn = Pipe()  # pipe for communication between models
    predictor1 = Predictor1(
        modelargs=args.model1,
        config=config1,
        queue_in=queue_main,
        queue_comm=queue_comm,
        pipe_end=reg_conn,
        detection_threshold=detection_thresh1,
        min_num_keypoints=min_num_keypoints1)
    predictor2 = Predictor2(
        modelargs=args.model2,
        config=config2,
        queue_in=queue_main,
        queue_comm=queue_comm,
        pipe_end=fast_conn,
        detection_threshold=detection_thresh2,
        min_num_keypoints=min_num_keypoints2)

    # start the processes
    # capture.start()
    predictor1.start()  # 1920x1080
    predictor2.start()  # 224x224

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

    # Stop the processes in order to exit the main program
    # capture.join()
    predictor1.join()
    predictor2.join()


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
