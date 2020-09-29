# -*- coding: utf-8 -*-

import numpy as np
from skimage import io, util
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize, rescale, rotate
from skimage.feature import hog

def random_rotation(img):
    """
    Rotate an image by a random angle

    Parameters
    ----------
    img : array-like of shape (size, size)
    """
    random_degree = np.random.randint(low = -25, high = 25)
    return rotate(img, random_degree)

def horizontal_flip(img):
    """
    Flip an image horitally

    Parameters
    ----------
    img : array-like of shape (size, size)
    """
    return img[:, ::-1]

def positive_sample(size, nb_img, labels, path):
    """
    Extract positiv examples and

    Parameters
    ----------
    size : integer
           Size of the squared labels and squared and positiv boxes to extract

    nb_img : integer
             Number of training images

    labels : array-like shape (nb_faces, 5)
             Labels of boxes which contains faces

    path : string
           Path to find training images

    Returns
    -------
    X_train : arra-like of shape (nb_examples, size, size)
              Extracted, rotated and flipped boxes

    squared_labels : array-like of shape (nb_faces, 5)
                     Squares labels of faces
    """
    X_train = np.empty([0, size, size])
    squared_labels = np.zeros(labels.shape)

    k = 0
    for img_idx in range(nb_img):
        image = io.imread(f'{path}/%04d.jpg'%(img_idx+1))
        # Keep only labels related to img_idx
        labels_filter = labels[labels[:,0] == (img_idx + 1)]
        for label in labels_filter:
            Ni, Nj = label[3:5]
            i, j = label[1:3]
            taille = int(max(Ni+6, Nj+6))
            i_prime = min(max(i-2 + Ni/2 - taille/2, 0), image.shape[0])
            j_prime = min(max(j-2 + Nj/2 - taille/2, 0), image.shape[1])
            squared_labels[k] = [(img_idx+1), i_prime, j_prime, taille, taille]
            box = image[int(i_prime):int(i_prime+taille), int(j_prime):int(j_prime+taille)]
            box = rgb2gray(box)
            box = resize(box, (size, size))
            X_train = np.append(X_train, [box], axis = 0)
            k += 1

    print('X_train apres création', X_train.shape)

    # Rotate and flip extracted boxes to create more positive examples
    n = X_train.shape[0]
    for img_idx in range(n):
        x_rot = random_rotation(X_train[img_idx])
        x_sym = horizontal_flip(X_train[img_idx])
        X_train = np.append(X_train, [x_rot], axis = 0)
        X_train = np.append(X_train, [x_sym], axis = 0)

    print('X_train augmenté', X_train.shape)
    return (X_train, squared_labels)

def sliding_window(img, patch_size, istep=5, jstep=5, scale=1.0):
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
            # Rescale width and height
            X_labels = np.append(X_labels, [[i*1/scale, j*1/scale, patch_size/scale, patch_size/scale]], axis=0)
    return (X, X_labels)

def add_negative_sample(X_train, nb_img, squared_labels, size, rand, path):
    """
    Extract negative examples

    Parameters
    ----------
    X_train : array-like of shape(n, size, size)
              Positiv boxes

    nb_img : integer
             Number of training images

    squared_labels : arra-like of shape (nb_faces, 5)

    size : integer
           Size of the squared negativ boxes to extract

    rand : integer
           Number of boxes to keep randomly

    path : string
           Path to find training images

    Returns
    -------
    X_train : arra-like of shape (n_examples, size, size)
              Positiv and negativ examples
    """
    for img_idx in range(nb_img):
        # Read the image img_idx
        image = io.imread(f'{path}/%04d.jpg'%(img_idx+1))
        image = rgb2gray(image)
        for scale in [2, 1.5, 1, 0.5]:
            # Retrieve negative examples from img_idx and associated labels
            box_neg, box_labels_neg = sliding_window(image, size, istep=100, jstep=100, scale=scale)

            # Keep only the positiv labels of the current image
            img_labels = squared_labels[squared_labels[:,0] == (img_idx+1)]

            # Create a mask to remove boxes whose IoU > 0,3
            mask_box = np.ones(box_labels_neg.shape[0], dtype = bool)
            for j in range(img_labels.shape[0]):
                for k in range(box_labels_neg.shape[0]):
                    if IoU(img_labels[j, 1:], box_labels_neg[k]) > 0.3:
                        mask_box[k] = False

            # Remove positive boxes
            box_neg = box_neg[mask_box]
            # Keep only rand random boxes from the image
            np.random.shuffle(box_neg)
            if box_neg.shape[0] > rand:
                box_neg = box_neg[:rand]

            if box_neg.shape[0] != 0:
                X_train = np.concatenate((X_train, box_neg))
    return X_train


def train_dataset(size, nb_img, labels, rand, path):
    """
    Create train examples

    Parameters
    ----------
    size : integer
           Size of the squared boxes to extract

    nb_img : integer
             Number of training images

    labels : arra-like of shape (nb_faces, 5)

    rand : integer
           Number of boxes to keep randomly when creating negativ examples

    path : string
           Path to find training images

    Returns
    -------
    X_train : arra-like of shape (nb_examples, size, size)
              Positiv and negativ examples

    y_train : array-like of shape (nb_examples,)
              Classes of boxes in X_train

    squared_labels : array-like of shape (nb_faces, 5)
    """
    # Create X_train
    X_train, squared_labels = positive_sample(size, nb_img, labels, path)
    y_train = np.ones(X_train.shape[0])
    X_train = add_negative_sample(X_train, nb_img, squared_labels, size, rand, path)

    # Create y_train
    print('X_train neg et pos', X_train.shape)
    y_train_neg = -np.ones(X_train.shape[0]-y_train.shape[0])
    y_train = np.concatenate((y_train, y_train_neg))
    print('y_train neg et pos', y_train.shape)

    # Calculate HOG descriptors
    X_train = np.array([hog(box) for box in X_train])
    print('X_train neg et pos avec hog', X_train.shape)

    return [X_train, y_train, squared_labels]

def best_classifier(size, X_train, y_train):
    """
    Validation to determine the best classifier

    Parameters
    ----------
    size : integer
           Size of the boxes

    X_train : arra-like of shape (nb_examples, size, size)
              Positiv and negativ examples

    y_train : array-like of shape (nb_examples,)
              Classes of boxes in X_train
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn import svm
    from sklearn import tree
    from sklearn.model_selection import train_test_split

    # split data in 2 groups : 70% of data for training and 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    print('split X_train', X_train.shape)
    print('split X_test', X_test.shape)
    print('split y_train', y_train.shape)
    print('split y_test', y_test.shape)

    N = np.array([5, 10, 20, 30, 40])
    err_KN = []
    err_ada = []
    err_forest = []
    for n in N:
        print('n', n)
        # KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors = n)
        clf.fit(X_train, y_train)
        err_KN.append(np.mean(clf.predict(X_test) != y_test))

        # AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators = n)
        clf.fit(X_train, y_train)
        err_ada.append(np.mean(clf.predict(X_test) != y_test))

        # RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = n)
        clf.fit(X_train, y_train)
        err_forest.append(np.mean(clf.predict(X_test) != y_test))

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    err_SVC = np.mean(clf.predict(X_test) != y_test)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    err_tree = np.mean(clf.predict(X_test) != y_test)

    print('err_KN', np.min(err_KN))
    print('err_ada', np.min(err_ada))
    print('err_forest', np.min(err_forest))
    print('err_SVC', err_SVC)
    print('err_tree', err_tree)

    n_min_KN = N[np.argsort(err_KN)]
    n_min_ada = N[np.argsort(err_ada)]
    n_min_forest = N[np.argsort(err_forest)]

    # barplot
    plt.bar([f"KN (n={n_min_KN[0]})", f"Ada (n={n_min_ada[0]})", f"Forest (n={n_min_forest[0]})", 'SVC', 'Tree'], [np.min(err_KN), np.min(err_ada), np.min(err_forest), err_SVC, err_tree])
    plt.title(f"Validation pour les boîtes de taille {size}")
    plt.show()

def SVC_validation(size, X_train, y_train):
    """
    Validation to determine parameters of SVC classifier

    Parameters
    ----------
    size : integer
           Size of the boxes

    X_train : arra-like of shape (nb_examples, size, size)
              Positiv and negativ examples

    y_train : array-like of shape (nb_examples,)
              Classes of boxes in X_train
    """
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)

    print('split X_train', X_train.shape)
    print('split X_test', X_test.shape)
    print('split y_train', y_train.shape)
    print('split y_test', y_test.shape)

    max_area_poly = 0
    max_area_rbf = 0
    max_area_sigmoid = 0
    max_area_linear = 0
    param_poly = [0, 0, 0]
    param_rbf = [0, 0]
    param_sigmoid = 0
    param_linear = 0
    for c in  [0.1, 1, 5, 10, 50]:
        print('C', c)
        # linear kernel
        linear_svc = svm.SVC(C=c, kernel='linear')
        linear_svc.fit(X_train, y_train)
        area_linear = roc_auc_score(y_test, linear_svc.decision_function(X_test))
        if area_linear > max_area_linear:
            max_area_linear = area_linear
            param_linear = c
        print('max_area_linear', max_area_linear)

        # polynomial kernel
        for d in [2, 3, 4, 5]:
            for g in [0.001, 0.01, 0.1, 1]:
                poly_svc = svm.SVC(C=c, degree = d, gamma = g, kernel='poly')
                poly_svc.fit(X_train, y_train)
                area_poly = roc_auc_score(y_test, poly_svc.decision_function(X_test))
                if area_poly > max_area_poly:
                    max_area_poly = area_poly
                    param_poly = [c, d, g]
        print('max_area_poly', max_area_poly)

        # rbf kernel
        for g in [0.001, 0.01, 0.1, 1]:
            rbf_svc = svm.SVC(C=c, gamma = g, kernel='rbf')
            rbf_svc.fit(X_train, y_train)
            area_rbf = roc_auc_score(y_test, rbf_svc.decision_function(X_test))
            if area_rbf > max_area_rbf:
                max_area_rbf = area_rbf
                param_rbf = [c, g]
        print('max_area_rbf', max_area_rbf)

        # sigmoid kernel
        sigmoid_svc = svm.SVC(C=c, kernel='sigmoid')
        sigmoid_svc.fit(X_train, y_train)
        area_sigmoid = roc_auc_score(y_test, sigmoid_svc.decision_function(X_test))
        if area_sigmoid > max_area_sigmoid:
            max_area_sigmoid = area_sigmoid
            param_sigmoid = c
        print('max_area_sigmoid', max_area_sigmoid)

    # barplot
    plt.bar([f"Linear (C={param_linear})", f"Poly (C={param_poly[0]}, degree={param_poly[1]}, gamma={param_poly[2]})",
             f"Rbf (C={param_rbf[0]}, gamma={param_rbf[1]})", f"sigmoid (C={param_sigmoid})"],
            [max_area_linear, max_area_poly, max_area_rbf, max_area_sigmoid])
    plt.title(f"Validation les paramètres des kernels du classifieur SVC")
    plt.show()

def validation_size_train(size, labels, nb_img):
    """
    Validation to determine the number of train examples

    Parameters
    ----------
    size : integer
           Size of the boxes

    labels : array-like shape (nb_faces, 5)
             Labels of boxes which contains faces

    nb_img : integer
             Number of training images

    Returns
    -------
    min : float
          Number of boxes which minimize the error rate
    """
    from sklearn.metrics import roc_auc_score
    from sklearn import svm
    from sklearn.model_selection import train_test_split

    err_min = 1
    min = 0
    for rand in [30, 35, 35, 40, 45, 45]:
        print('rand', rand)
        X_train, y_train, squared_labels = train_dataset(size, nb_img, labels, rand)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=10000)

        clf = svm.SVC(C=1, kernel='poly', degree=4, gamma=0.1)
        clf.fit(X_train, y_train)

        err = np.mean(clf.predict(X_test) != y_test)
        if err < err_min:
            err_min = err
            min = X_train.shape[0]
        print('pour X_train', X_train.shape, 'taux d erreur', err)

    return min

def IoU(box1, box2):
    """
    Calculate intersection over union

    Parameters
    ----------
    box1 : array-like of shape (4,)
           First box

    box2 : array-like of shape (4,)
           Second box
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

def first_detection(size, nb_img, X_train, y_train, squared_labels, clf, path):
    """
    Fisrt detections on training images to add false positiv to train examples

    Parameters
    ----------
    size : integer
           Size of the boxes

    nb_img : integer
             Number of training images

    X_train : arra-like of shape (nb_examples, size, size)
              Positiv and negativ examples

    y_train : array-like of shape (nb_examples,)
              Classes of boxes in X_train

    squared_labels : array-like of shape (nb_faces, 5)

    clf : object
          Classifier

    path : string
           Path to find training images

    Returns
    -------
    X_train : arra-like of shape (nb_examples, size, size)
              False positiv, positiv and negativ examples

    y_train : array-like of shape (nb_examples,)
              Classes of boxes in X_train
    """
    for img_idx in range(nb_img):
        # Read the image img_idx
        image = io.imread(f'{path}/%04d.jpg'%(img_idx+1))
        image = rgb2gray(image)
        print('img_idx', (img_idx+1))
        for scale in [1.25, 1, 0.75, 0.5]:
            box_test, box_labels_test = sliding_window(image, size, istep=7, jstep=7, scale=scale)
            img_labels = squared_labels[squared_labels[:,0] == (img_idx+1)]

            nb_box = box_labels_test.shape[0]
            y_box_test = -np.ones(nb_box)

            for j in range(img_labels.shape[0]):
                for k in range(nb_box):
                    if IoU(img_labels[j, 1:], box_labels_test[k]) > 0.40:
                        y_box_test[k] = 1

            # Calculate descriptors got by sliding window
            box_test = np.array([hog(box) for box in box_test])
            y_predict = clf.predict(box_test)

            # Create a mask to keep false positiv boxes
            mask_neg = np.array([(y_test == -1 and y == 1) for y, y_test in zip(y_predict, y_box_test)])
            box_test = box_test[mask_neg]

            # If there is false positiv, we add them to the train examples
            n = box_test.shape[0]
            if n != 0:
                X_train = np.concatenate((X_train, box_test))
                y_train = np.concatenate((y_train, -np.ones(n)))
                print('new X_train for scale', scale, ':', X_train.shape)

    return (X_train, y_train)

labels = np.loadtxt('./project_train/label_train.txt')
nb_img = 1000
size = 48
path = './project_train/train'

X_train, y_train, squared_labels = train_dataset(size, nb_img, labels, 15, path)

# Learn a first classifier
from sklearn import svm
clf = svm.SVC(C=1, kernel='poly', degree=4, gamma=0.1)
clf.fit(X_train, y_train)

X_train, y_train = first_detection(size, nb_img, X_train, y_train, squared_labels, clf, path)

print('X_train avec predict', X_train.shape)
print('y_train avec predict', y_train.shape)

# Learn a second classifier
clf_new = svm.SVC(C=1, kernel='poly', degree=4, gamma=0.1)
clf_new.fit(X_train, y_train)

import pickle5 as pickle
# Save the model
filename = 'svm_model.sav'
pickle.dump(clf_new, open(filename, 'wb'))
