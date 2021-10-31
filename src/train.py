
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

import yaml
import argparse
from tensorflow.keras.layers import  Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from glob import glob
import get_data
import json

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_param_filename(config_path):
    config = read_params(config_path)
    param_filepath = config["reports"]["params"]
    return param_filepath

def get_hyperparam(config_path):
    config = read_params(config_path)
    loss = config["estimators"]["Transer_learning"]["params"]["loss"]
    optimizer = config["estimators"]["Transer_learning"]["params"]["optimizer"]
    epochs = config["estimators"]["Transer_learning"]["params"]["epochs"]
    batch_size = config["estimators"]["Transer_learning"]["params"]["batch_size"]
    target_size = config["estimators"]["Transer_learning"]["params"]["target_size"]
    model_name = config["estimators"]["Transer_learning"]["params"]["model"]
    return loss, optimizer, epochs, batch_size, target_size, model_name

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train,cv,test = get_data.main()
    loss, optimizer, epochs, batch_size, target_size, model_name = get_hyperparam(config_path=parsed_args.config)
    params_file = get_param_filename(config_path=parsed_args.config)
    IMAGE_SIZE = [target_size, target_size,3]
    if model_name == 'VGG16':
        vgg16 = VGG16(input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)
        for layer in vgg16.layers:
            layer.trainable = False
        folders = glob('data_given/train/*')
        x = Flatten()(vgg16.output)
        prediction = Dense(len(folders), activation='softmax')(x)
        model = Model(inputs=vgg16.input, outputs=prediction)

    if model_name == 'RESNET50':
        resnet50 = ResNet50(input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)
        for layer in resnet50.layers:
            layer.trainable = False
        folders = glob('data_given/train/*')
        x = Flatten()(resnet50.output)
        prediction = Dense(len(folders), activation='softmax')(x)
        model = Model(inputs=resnet50.input, outputs=prediction)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    vgg_mod = model.fit_generator(
        train,
        validation_data=cv,
        epochs=epochs,
        steps_per_epoch=len(cv),
        validation_steps=len(test)
    )

    model.save('saved_models/model.h5')

    with open(params_file,"w") as f:
        params = {
            "loss ": loss,
            "optimizer ": optimizer,
            "model ": model_name,
            "epochs ": epochs,
            "batch_size ": batch_size,
            "target_size ": target_size
        }
        json.dump(params, f, indent=4)




