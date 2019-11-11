import os
import glob
import itertools
import cv2

import shutil


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def parse_size(text):
    w, h = text.split('x')
    w = float(w)
    h = float(h)
    if w.is_integer():
        w = int(w)
    if h.is_integer():
        h = int(h)
    return w, h


def parse_kwargs(args):
    if args == '':
        return {}

    kwargs = {}
    for arg in args.split(','):
        key, value = arg.split('=')
        kwargs[key] = value

    return kwargs


def save_files(result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir, 'src'))
    result_src_dir = os.path.join(result_dir, 'src')
    file_list = glob.glob('*.py') + glob.glob('*.sh') + glob.glob('*.ini')
    file_list = file_list + glob.glob('*.tsv') + glob.glob('*.txt') + glob.glob("*.ipynb")
    for file in file_list:
        shutil.copy(file, os.path.join(result_src_dir, os.path.basename(file)))
    return result_src_dir


def show_image1(path, scale=500, name='UNK'):
    name = 'scale: ' + str(round(scale, 2))+' image: ' + name
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, 40, 30)
    cv2.resizeWindow(name, (480, 270))
    img = cv2.imread(path)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image2(image, name='UNK'):
    name = "test"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, 40, 30)
    #cv2.resizeWindow(name, (480, 270))
    #img = cv2.imread(path)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


