import os
import csv
import subprocess
from sklearn.model_selection import train_test_split

DATASET_DIR = './driving_dataset'


def split_train_val(csv_driving_data, test_size=0.1):
    """
    Splits the csv containing driving data into training and validation
    :param csv_driving_data: file path of csv driving data
    :return: train_split, validation_split
    """
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        driving_data = [row for row in reader][1:]

    train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=1)

    return train_data, val_data


def prepare_driving_dataset(cfg):
    if not os.path.isdir(cfg.DATASETS.DATA_DIR):
        # Download data from google drive and unzip to `./driving_dataset`
        subprocess.call('bash ./download_dataset.sh', shell=True)
    # split annotation files
    annotation_file = os.path.join(DATASET_DIR, './data.txt')
    train_list, val_list = split_train_val(annotation_file)
    # dump lists to csv
    with open(cfg.DATASETS.ANN_TRAIN_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_list)
    with open(cfg.DATASETS.ANN_VAL_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(val_list)
    vis_list = val_list[:10]
    with open(cfg.DATASETS.ANN_VIS_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vis_list)
