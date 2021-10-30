
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
from glob import glob
import get_data


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

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train,cv,test = get_data.main()
    loss, optimizer, epochs, batch_size, target_size = get_hyperparam(config_path=parsed_args.config)
    IMAGE_SIZE = [target_size, target_size,3]
    vgg16 = VGG16(input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)
    for layer in vgg16.layers:
        layer.trainable = False
    folders = glob('data_given/train/*')
    x = Flatten()(vgg16.output)
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=vgg16.input, outputs=prediction)

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

    model.save('saved_models/model_vgg16.h5')



