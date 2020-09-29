import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DIRDATA = "C:/Users/otisj/Machine Learning/Emotions project" # the actual path of the images
CATAGORIES = ["happy", "sad"] #indexes the two different types of data

for catagory in CATAGORIES:
    path = os.path.join(DIRDATA, catagory) # this gets us to the path of the images
    for image in os.listdir(path):
        # IMREAD_GRAYSCALE turns the image gray in order to reduce bias
        imag_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE) #creates a new array of all the images in the directories
        break
    break

image_size = 95 # sets all the images to the same size
new_array = cv2.resize(imag_array, (image_size, image_size)) #

training_data = []


def generate_training_data():
    for catagory in CATAGORIES:
        path = os.path.join(DIRDATA, catagory)  # this gets us to the path of the images
        class_number = CATAGORIES.index(catagory)
        # this assigns the images a number depending on their index in the catagories list
        for image in os.listdir(path):
            # IMREAD_GRAYSCALE turns the image gray in order to reduce bias
            imag_array = cv2.imread(os.path.join(path, image),
                                    cv2.IMREAD_GRAYSCALE)  # creates a new array of all the images in the directories
            new_array = cv2.resize(imag_array, (image_size, image_size))
            training_data.append([new_array, class_number])


generate_training_data()

print(len(training_data)) # this shows how much data is in each set (if they are not equal there may be some bias)

random.shuffle(training_data)

for sample in training_data: # check if the labels are correct
    print(sample[1])

training_set = [] # feature set (training)
test_set = [] # test set (testing)

for features, label in training_data:  # sorts the contents of training data into two arrays
    training_set.append(features) # this array gets the input images
    test_set.append(label) # this array gets the predicted output to the inputs in training_set.

training_set = np.array(training_set).reshape(-1, image_size, image_size, 1)  # must make training set a numpy array

test_set = np.array(test_set)
save_train = open("training_data.pickle", "wb")
pickle.dump(training_set, save_train)
save_train.close()

save_test = open("testing_data.pickle", "wb")
pickle.dump(test_set, save_test)
save_test.close()

