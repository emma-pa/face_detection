# -*- coding: utf-8 -*-

import numpy as np
from skimage import io, util
from sklearn import svm
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
from skimage.feature import hog
import pickle5 as pickle

def sliding_window(img, patch_size, istep=2, jstep=2, scale=1.0):
    """
    Slide a window to extract boxes from an image

    Parameters
    ----------
    img : array-like of various shape
         Image where patches are extracted

    patch_size : integer
                 Size of patches to extract

    istep : integer
            Steps of the window along the horizontal axis

    jstep : integer
            Steps of the window along the vertical axis

    scale : float
            Scale to rescale the image

    Returns
    -------
    X : array-like of shape (nb_patches, patch_size, patch_size)
        Extracted patches


    X_labels : array-like of shape (nb_patches, 4)
               Labels of the extracted patches
    """
    img = rescale(img, scale)
    X = np.empty([0, patch_size, patch_size])
    X_labels = np.empty([0, 4])
    for i in range(0, img.shape[0] - patch_size, istep):
        for j in range(0, img.shape[1] - patch_size, jstep):
            patch = img[int(i):int(i + patch_size), int(j):int(j + patch_size)]
            X = np.append(X, [patch], axis=0)
            # on remet à échelle afin de pouvoir comparer avec les autres boites
            X_labels = np.append(X_labels, [[i*1/scale, j*1/scale, patch_size/scale, patch_size/scale]], axis=0)
    return (X, X_labels)

def IoU(box1, box2):
    """
    Calculate intersection over union

    Parameters
    ----------
    box1 : array-like of shape (4,)
           Size of the squared negativ boxes to extract

    box2 : array-like of shape (4,)
           Size of the squared negativ boxes to extract
    """
    i1, j1, h1, l1 = box1
    i2, j2, h2, l2 = box2

    l_intersection = min(j1 + l1, j2 + l2) - max(j1, j2)
    h_intersection = min(i1 + h1, i2 + h2) - max(i1, i2)

    if l_intersection <= 0 or h_intersection <= 0:
        return 0

    I = l_intersection * h_intersection

    U = l1 * h1 + l2 * h2 - I

    return I / U

def build_detection(size, nb_img, clf, path):
    """
    Extract boxes and predict their class to build detections set

    Parameters
    ----------
    size : integer
           Size of the boxes to extract

    nb_img : integer
             Number of test images

    clf : object
          Classifier

    path : string
           Path to find test images

    Returns
    -------
    detections : array-like shape (nb_detections, 6)
                 Labels of detections supposed to be faces
    """
    detections = np.empty([0, 6])
    for img_idx in range(nb_img):
        print('img_idx', img_idx+1)
        image = io.imread(f'{path}/%04d.jpg'%(img_idx+1))
        image = rgb2gray(image)
        for scale in [1.25, 1, 0.75, 0.5, 0.4, 0.35, 0.30, 0.35, 0.2, 0.15, 0.1, 0.05]:
            if (np.floor(scale*image.shape[0]) > size and np.floor(scale*image.shape[1]) > size) :
                box_test, box_labels_test = sliding_window(image, size, istep=7, jstep=7, scale=scale)
                box_test = [hog(box) for box in box_test]
                y =  clf.predict(box_test)

                # Calculate scores
                score = np.absolute(clf.decision_function(box_test))

                # Keep only positives detections
                mask_detection = y[:] == 1
                box_labels_test = box_labels_test[mask_detection]
                score = score[mask_detection]

                for box, s in zip(box_labels_test, score):
                    i, j, Ni, Nj = box
                    detections = np.append(detections, [[(img_idx+1), i, j, Ni, Nj, s]], axis=0)

    return detections

def remove_non_maxima(detections):
    """
    Remove non maxima detections

    Parameters
    ----------
    detections : array-like shape (nb_detections, 6)
                 Labels of detections supposed to be faces

    Returns
    -------
    results : array-like shape (nb_maxima, 6)

    """
    results = np.array([])
    i = 0
    while i in range(0, detections.shape[0]):
        img_idx = detections[i, 0]
        mask_img = detections[:,0] == img_idx
        img_detections = detections[mask_img]

        # Order scores
        img_detections = np.array(sorted(img_detections, key = lambda box: box[5], reverse = True))

        nb_detection = img_detections.shape[0]
        mask_box = np.ones(nb_detection, dtype = bool)
        for j in range(nb_detection):
            box1 = img_detections[j, 1:5]
            for k in range(j+1, nb_detection):
                box2 = img_detections[k, 1:5]
                if IoU(box1, box2) > 0.5:
                    mask_box[k] = False
        img_detections = img_detections[mask_box]

        if results.size == 0:
            results = img_detections
        else:
            results = np.concatenate((results, img_detections))
        i += nb_detection

    return results

# Load the model
filename = 'svm_model.sav'
clf = pickle.load(open(filename, 'rb'))

path = './project_test/test'
size = 48
nb_img = 500
detections = build_detection(size, nb_img, clf, path)
detections = remove_non_maxima(detections)

# Save detections
np.savetxt('detection.txt', detections , fmt = '%f')
