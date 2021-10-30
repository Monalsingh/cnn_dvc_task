## read params
## process
## return dataframe
import os
import yaml
import pandas as pd
import argparse
from keras.preprocessing.image import ImageDataGenerator

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data_path(config_path):
    config = read_params(config_path)
    # print(config)
    data_path = []
    train_data_path = config["data_source"]["s3_source_train"]
    data_path.append(train_data_path)
    test_data_path = config["data_source"]["s3_source_test"]
    data_path.append(test_data_path)
    return data_path

def our_preprocessing_function(filename):
    #Combines all the transformations
    img = cv2.imread(filename)
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe2.apply(img)
    cv2.imwrite('tmp/clahe.jpg', image)
    img = cv2.imread("tmp/clahe.jpg")
    return img

def preprocess_data():
    img_datagen = ImageDataGenerator(rescale=1. / 255,
                                     preprocessing_function=our_preprocessing_function)
    return img_datagen


def get_data(data_path):
    img_datagen = preprocess_data()
    train_it = img_datagen.flow_from_directory(data_path[0], class_mode='categorical')
    test_it = img_datagen.flow_from_directory(data_path[1], class_mode='categorical')
    return train_it, test_it


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data_path = get_data_path(config_path=parsed_args.config)
    train, test = get_data(data_path)
    print(train)
    print(test)