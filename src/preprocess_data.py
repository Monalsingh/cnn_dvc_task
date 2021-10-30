from keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
from matplotlib import cm
import numpy as np
import argparse
import yaml

def our_preprocessing_function(filename):
    #Combines all the transformations
    print(filename)
    img = filename
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #image = clahe2.apply(img)
    image = clahe2.apply(gray)
    cv2.imwrite('tmp/clahe.jpg', image)
    img = cv2.imread("tmp/clahe.jpg")
    return img
'''
def preprocess_data():
    img_datagen_train = ImageDataGenerator(rescale=1. / 255, preprocessing_function=our_preprocessing_function, validation_split=0.2)
    return img_datagen_train

def preprocess_data1():
    img_datagen_test = ImageDataGenerator(rescale=1. / 255, preprocessing_function=our_preprocessing_function)
    return img_datagen_test
'''

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def preprocess_data():
    img_datagen_train = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    return img_datagen_train

def preprocess_data1():
    img_datagen_test = ImageDataGenerator(rescale=1. / 255)
    return img_datagen_test

def get_info(config_path):
    config = read_params(config_path)
    target_size = config["estimators"]["VGG_transer_learning"]["params"]["target_size"]
    return target_size


def get_data(data_path):
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    target_size = get_info(config_path=parsed_args.config)
    img_datagen_train = preprocess_data()
    img_datagen_test = preprocess_data1()
    train_it = img_datagen_train.flow_from_directory(data_path[0],target_size = (target_size, target_size), class_mode='categorical', subset='training')
    cv_it = img_datagen_train.flow_from_directory(data_path[0], target_size = (target_size, target_size),class_mode='categorical', subset='validation')
    test_it = img_datagen_test.flow_from_directory(data_path[1], target_size = (target_size, target_size),class_mode='categorical')
    return train_it,cv_it,test_it