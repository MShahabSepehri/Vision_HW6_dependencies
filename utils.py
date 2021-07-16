import numpy as np
import cv2 as cv
import tensorflow as tf
from random import shuffle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
from Vision_HW6_dependencies import augmentation_utils as au


def get_image(path, size):
    image = cv.imread(path)
    image = cv.resize(image, size, interpolation=cv.INTER_AREA)
    return image


def create_label_dic(path):
    label_dic = {}
    num_cls = 0
    for subdir, _, _ in os.walk(path):
        cls = subdir.replace(path, '')
        if cls == '':
            continue
        if cls in label_dic.keys():
            continue
        label_dic[cls] = num_cls
        num_cls += 1

    label_reverse_dic = {}
    for key in label_dic.keys():
        label_reverse_dic[label_dic[key]] = key
    return label_dic, label_reverse_dic


def convert_to_oneHot(label, num_cls, eps=0):
    one_hot_label = np.zeros(num_cls) + eps
    one_hot_label[label] = 1 - (num_cls - 1) * eps
    return one_hot_label


def create_data(path, label_dic, size=(227, 227), augment=False):
    data = []
    labels = []
    num_cls = len(label_dic.keys())
    for subdir, _, files in os.walk(path):
        for file in files:
            image = get_image(os.path.join(subdir, file), size)
            data.append(tf.image.per_image_standardization(image))
            label = label_dic[subdir.replace(path, '')]
            labels.append(convert_to_oneHot(label, num_cls))

    if augment:
        data, labels = au.augment_data(data, labels)

    pack = list(zip(data, labels))
    pack = shuffle(pack)
    data, labels = zip(*pack)

    return np.array(data).astype(np.float32), np.array(labels).astype(np.float32)


def get_label_str(label_reverse_dic, label):
    return label_reverse_dic[np.argmax(label)]


def plot_results(history):
    plt.figure(figsize=(5, 5))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['train', 'val'])
    plt.title('Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'])
    plt.title('Loss')
    plt.tight_layout()
    plt.show()
