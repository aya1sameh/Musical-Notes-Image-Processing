import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np
import argparse
import os
import random
from sklearn.model_selection import train_test_split
import pickle
import math


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


classifiers = {
    'SVM': svm.LinearSVC(random_state=random_seed, max_iter=20000,dual=False),
}



def extract_hog_features(img, target_img_size=(32, 32)):

    img = cv.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def extract_hist_features(img, target_img_size=(32, 32)):

    if(len(img.shape)==3):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, target_img_size)
    img = img > 127
    hist_horizontal = np.sum(img, axis=0)
    hist_vertical = np.sum(img, axis=1)
    return np.concatenate((hist_vertical, hist_horizontal))

def extract_huMoments_features(img, target_img_size = (32,32)):
    
    if(len(img.shape)==3):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, target_img_size)
    _,img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
    Moments = cv.moments(img)
    huMoments = cv.HuMoments(Moments)
    for i in range(0,7):
        if huMoments[i] != 0:
            huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
    return huMoments.flatten()

def extract_features(img):
    aspectRatio = img.shape[0] / img.shape[1]

    hogFeatures = extract_hog_features(img)
    histFeatures = extract_hist_features(img)
    MomentsFeatures = extract_huMoments_features(img)

    allFeatures = np.append(hogFeatures, aspectRatio)
    allFeatures = np.concatenate((allFeatures, histFeatures))
    allFeatures = np.concatenate((allFeatures,MomentsFeatures))
    return allFeatures


def load_dataset(path_to_dataset):
    features = []
    labels = []
    path_to_dataset = os.path.join(os.getcwd(), path_to_dataset)
    FoldersNames = os.listdir(path_to_dataset)
    print(FoldersNames)
    for Folder in FoldersNames:
        print(Folder)
        img_filenames = os.listdir(os.path.join(path_to_dataset, Folder))
        for i, fn in enumerate(img_filenames):

            labels.append(Folder)

            path = os.path.join(path_to_dataset, Folder, fn)
            img = cv.imread(path)
            features.append(extract_features(img))

            # show an update every 1,000 images
            if i > 0 and i % 500 == 0:
                print("[INFO] processed {}/{}".format(i, len(img_filenames)))

    return features, labels


def train_classifier(path_to_dataset):

    # Load dataset with extracted features
    print('Loading dataset. This will take time ...')
    features, labels = load_dataset(path_to_dataset)
    print('Finished loading dataset.')

    # Since we don't want to know the performance of our classifier on images it has seen before
    # we are going to withhold some images that we will test the classifier on after training
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed, stratify=labels, shuffle=True)

    print('############## Training', " SVM ", "##############")
    # Train the model only on the training features
    model = classifiers['SVM']
    model.fit(train_features, train_labels)

    # Test the model on images it hasn't seen before
    accuracy = model.score(test_features, test_labels)

    print("SVM ", 'accuracy:', accuracy*100, '%')


def main():
    train_classifier("Dataset")
    classifier = classifiers['SVM']
    filename = 'Test.sav'
    pickle.dump(classifier, open(filename, 'wb'))


if __name__ == "__main__":
    main()
    