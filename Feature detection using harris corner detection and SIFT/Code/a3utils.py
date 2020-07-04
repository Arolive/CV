# Please do not modify this file.
import numpy as np
import cv2
import pickle


def im2single(im):
    im = im.astype(np.float32) / 255

    return im

def single2im(im):
    im *= 255
    im = im.astype(np.uint8)

    return im

def rgb2gray(rgb):
    """Convert RGB image to grayscale
    Args:
    - rgb: A numpy array of shape (m,n,c) representing an RGB image
    Returns:
    - gray: A numpy array of shape (m,n) representing the corresponding grayscale image
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def load_image(path):
    """
    Args:
    - path: string representing a filepath to an image
    """
    return im2single(cv2.imread(path))[:, :, ::-1]

def save_image(path, im):
    """
    Args:
    - path:
    - im: A numpy array of shape
    """
    return cv2.imwrite(path, single2im(im.copy())[:, :, ::-1])

def cheat_interest_points(eval_file, scale_factor):
    """
    This function is provided for development and debugging but cannot be used in
    the final handin. It 'cheats' by generating interest points from known
    sources. It will only work for the 3 images provided in the /data subdirectory.

    Args:
    - eval_file: string representing the file path to the list of known correspondences
    - scale_factor: Python float representing the scale needed to map from the original
            image coordinates to the resolution being used for the current experiment.

    Returns:
    - x: A numpy array of shape (k,) containing ground truth x-coordinates of imgA correspondence pts
    - y: A numpy array of shape (k,) containing ground truth y-coordinates of imgA correspondence pts
   """
    with open(eval_file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')

    return d['x1'] * scale_factor, d['y1'] * scale_factor

def show_interest_points(img, X, Y):
    """
    Visualized interest points on an image with random colors

    Args:
    - img: A numpy array of shape (M,N,C)
    - X: A numpy array of shape (k,) containing x-locations of interest points
    - Y: A numpy array of shape (k,) containing y-locations of interest points

    Returns:
    - newImg: A numpy array of shape (M,N,C) showing the original image with
            colored circles at keypoints plotted on top of it
    """
    newImg = img.copy()
    for x, y in zip(X.astype(int), Y.astype(int)):
        cur_color = np.random.rand(3)
        newImg = cv2.circle(newImg, (x, y), 4, cur_color, -1, cv2.LINE_AA)

    return newImg
