"""
- Takes a CSV of all data
- Splits it into train-test in SPLIT_RATIOS (ratios are for each class)
- Writes the modified CSV to filenames in SPLIT_CSV_NAMES
- Moves the second and later part of splits from the directory with all data, to a new directory, specified by SPLIT_DIR_NAMES

CSVs and Data are assumed to be in directories specified in `config`
"""

import os
import pandas as pd

from .config import config 

SPLIT_RATIOS = [0.25]
SPLIT_DIR_NAMES = ["test"]
SPLIT_CSV_NAMES = ["ISIC18_T3_test.csv"]

SRC_CSV_NAME = "ISIC18_T3_train.csv"
SRC_DIR_NAME = "train"

DATA_ROOT_PATH = config.isic18_t3_root_path

assert len(SPLIT_RATIOS) == len(SPLIT_DIR_NAMES) == len(SPLIT_CSV_NAMES), "Split parameter lists must be of same length"


def split_data():

    # Fetch CSV
    alldata_csv_path = os.path.join(
        config.csv_root_path,
        SRC_CSV_NAME
    )
    assert os.path.isfile(alldata_csv_path), f"CSV file was not found at {alldata_csv_path}"

    # Store CSV as Dataframe
    alldata_csv_df = pd.read_csv(alldata_csv_path)

    # Ensure data path validity
    alldata_img_path = os.path.join(
        DATA_ROOT_PATH,
        SRC_DIR_NAME
    )
    assert os.path.isdir(alldata_img_path), f"Need valid data path as `root`. Got {alldata_img_path}"

    acc = 0
    split_dfs = [ 
        pd.DataFrame(columns=alldata_csv_df.columns)
        for _ in range(len(SPLIT_RATIOS)) 
    ]
    # Split and concatenate for each class
    for idx, classname in enumerate(alldata_csv_df.columns):
        
        # get all rows for current class
        class_rows_df = alldata_csv_df.loc[alldata_csv_df[classname]==1.0, :]
       
        # make all the splits

    print(acc)

    num_classes = len(class_id_map)
    class_names = list(class_id_map.keys())
    # All `target` values of the dataset
    labels = list(map(
        lambda idx: get_sparse_label(csv_df, idx),
        range(len(csv_df))
    ))

