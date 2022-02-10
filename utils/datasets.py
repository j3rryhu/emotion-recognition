import numpy as np
import cv2
import os
from keras.utils import to_categorical


class DatasetLoader(object):
    def __init__(self, dataset_path, image_size=(48, 48)):

        self.dataset_name = 'fer2013'
        self.dataset_path = dataset_path
        self.image_size = image_size

    def load(self, usage):
        if usage=='train':
            faces = []
            labels = []
            trainingset_path = self.dataset_path + '/Training'
            categories = os.listdir(trainingset_path)
            for label in range(0, len(categories)):
                category_path = os.path.join(trainingset_path, str(label))
                imgs = os.listdir(category_path)
                for img in imgs:
                    img_path = os.path.join(category_path, img)
                    face = cv2.imread(img_path)
                    face = np.array(face)
                    faces.append(face)
                    label = to_categorical(label, 7)
                    labels.append(label)
            faces = np.vstack(faces)
            labels = np.vstack(labels)
            return faces, labels
        if usage=='test':
            faces = []
            labels = []
            testset_path = self.dataset_path + '/Private Test'
            categories = os.listdir(testset_path)
            for label in range(0, len(categories)):
                category_path = os.path.join(testset_path, str(label))
                imgs = os.listdir(category_path)
                for img in imgs:
                    img_path = os.path.join(category_path, img)
                    face = cv2.imread(img_path)
                    face = np.array(face)
                    faces.append(face)
                    label = to_categorical(label, 7)
                    labels.append(label)
            faces = np.vstack(faces)
            labels = np.vstack(labels)
            return faces, labels


    def get_labels(self):
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}





