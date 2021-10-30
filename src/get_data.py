## read params
## process
## return dataframe
import os
import yaml
import pandas as pd
import argparse
import preprocess_data as preprocess
import pickle

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

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data_path = get_data_path(config_path=parsed_args.config)
    train, test = preprocess.get_data(data_path)
    print(train)
    print(test)
    with open(os.path.join(dir_, "img.csv"), "w") as f:
        pass