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
DATA_EXTN = "jpg"

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


    split_dfs = [ 
        pd.DataFrame(columns=alldata_csv_df.columns)
        for _ in range(len(SPLIT_RATIOS)) 
    ]
    # Split and concatenate for each class
    for classname in alldata_csv_df.columns:
        
        # get all rows for current class
        class_rows_df = alldata_csv_df.loc[alldata_csv_df[classname]==1.0, :]
        num_rows = len(class_rows_df)
       
        # make all the splits
        for idx, ratio in enumerate(SPLIT_RATIOS):

            # sample rows randomly, and append to split
            rows_for_split = class_rows_df.sample(
                n = int(num_rows * ratio)
            )
            split_dfs[idx] = split_dfs[idx].append(
                rows_for_split, 
                ignore_index=True
            )
            
            # drop split rows from main df
            alldata_csv_df = alldata_csv_df.drop(rows_for_split.index)

    # Save split dataframes
    for idx, name in enumerate(SPLIT_CSV_NAMES):
        split_dfs[idx].to_csv(
            os.path.join(
                config.csv_root_path,
                name
            ),
            index=False
        )
        print(f"Dataframe with {len(split_dfs[idx])} images saved, for split name {name}")
    
    # Save the modified original df
    alldata_csv_df.to_csv(alldata_csv_path, index=False)

    
    # Move corresponding data for splits
    for idx, dir_name in enumerate(SPLIT_DIR_NAMES):

        # create destination if it doesn't exist
        destn_path = os.path.join(
            DATA_ROOT_PATH,
            dir_name 
        )
        if not os.path.isdir(destn_path):
            Path(destn_path).mkdir(
                parents=True,
                exist_ok=False
            )

        # move split's images
        for img_name in split_dfs[idx].iloc[:, 0]:
            os.rename(
                os.path.join(
                    alldata_img_path,
                    f"{img_name}.{DATA_EXTN}"
                ),
                os.path.join(
                    destn_path,
                    f"{img_name}.{DATA_EXTN}"
                )
            )
    
    print("Images for splits moved")




