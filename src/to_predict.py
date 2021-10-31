from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from keras.applications.vgg16 import decode_predictions
import numpy as np
import get_data
import os
import yaml
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_hyperparam(config_path):
    config = read_params(config_path)
    loss = config["estimators"]["VGG_transer_learning"]["params"]["loss"]
    optimizer = config["estimators"]["VGG_transer_learning"]["params"]["optimizer"]
    epochs = config["estimators"]["VGG_transer_learning"]["params"]["epochs"]
    batch_size = config["estimators"]["VGG_transer_learning"]["params"]["batch_size"]
    target_size = config["estimators"]["VGG_transer_learning"]["params"]["target_size"]
    return loss, optimizer, epochs, batch_size, target_size


def test_data_seperator(test_generator):
    data = []     # store all the generated data batches
    labels = []   # store all the generated label batches
    max_iter = 10  # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
    i = 0
    for d, l in test_generator:
        data.append(d)
        labels.append(l)
        #i += 1
        #if i == max_iter:
        #    break
    return data, labels

def data_label_formatter(data, labels):
    data = np.array(data, dtype=object)
    print(data.shape)
    data = np.reshape(data, (data.shape[0] * data.shape[1],) + data.shape[2:])

    labels = np.array(labels, dtype=object)
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1],) + labels.shape[2:])

    return data, labels

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    loss, optimizer, epochs, batch_size, target_size = get_hyperparam(config_path=parsed_args.config)
    model = load_model('saved_models/model_vgg16.h5')
    print("model loaded.......")
    _, _, test = get_data.main()
    #print(test)
    #data, labels = test_data_seperator(test)
    #data, labels = data_label_formatter(data, labels)
    #print(data.shape)
    #scores = model.evaluate(data[0:3], labels[0:3], verbose=0)
    scores = model.evaluate(test, verbose=0)
    print(scores)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("===========================================================")
    print("Evaluating model on single test image....")
    #img = image.load_img('data_given/test/covid/covid_19_190.png', target_size=(128, 128))
    #img = image.load_img('data_given/test/normal/normal_1312.png', target_size=(128, 128))
    img = image.load_img('data_given/train/normal/normal_020.png', target_size=(128, 128))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    print("image_data shape: " + str(img_data.shape))
    yhat = model.predict(img_data)
    print(yhat)
    yhat_class = np.argmax(model.predict(img_data), axis=1)
    print(yhat_class)
    if yhat_class == 0:
        print("covid")
    if yhat_class == 1:
        print("normal")
    if yhat_class == 2:
        print("viral")

    dir_path = 'evaluation/report'
    filename = dir_path + '.txt'
    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    f = open(filename, append_write)
    f.write("Time: " + datetime.now().strftime('%Y%m%d_%H%M') + '\n')
    f.write("Evaluation on test set details...."+'\n')
    f.write("loss function used: "+str(loss)+'\n')
    f.write("Optimizer used: " + str(optimizer)+'\n')
    f.write("Epochs used: " + str(epochs)+'\n')
    f.write("Batch size: " + str(batch_size)+'\n')
    f.write("Image target size: "+str(target_size)+'\n')
    f.write("test data accuracy" + str(scores[1] * 100)+'\n')
    f.write("=======================================================" + '\n')
    f.close()


    #label = decode_predictions(yhat)
    #label = label[0][0]
    #print('%s (%.2f%%)' % (label[1], label[2] * 100))

    # convert the probabilities to class labels






