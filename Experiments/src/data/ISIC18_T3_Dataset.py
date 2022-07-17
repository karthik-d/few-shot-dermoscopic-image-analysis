from torch.utils.data import Dataset
from torchvision import io 
import os 
import pandas as pd
import torch

from .config import config

class ISIC18_T3_Dataset(Dataset):

    """
    Extended custom `Dataset` class to interface
    torch with the ISIC-2018 T3 Dataset
    """

    class_id_map = dict(
        MEL = 0,
        NV = 1,
        BCC = 2,
        AKIEC = 3,
        BKL = 4,
        DF = 5,
        VASC = 6
    )


    def __init__(self, root, mode='train', img_ext='jpg', transform=None, target_transform=None):
        
        """
        Initialize the Dataset
        - root: Root directory containing all data files
        - mode: Determines the CSV file to use (ISIC18_T3_<mode>.csv)
                Also determined the subdirectory for data split
                Valid values: <subdirectory names within the data root>

        NOTE: CSV must be present in the [ROOT]/data directory
        """
        
        super(ISIC18_T3_Dataset, self).__init__()

        # Fetch CSV
        self.csv_path = os.path.join(
            config.csv_root_path,
            f"ISIC18_T3_{mode}.csv"
        )
        assert os.path.isfile(self.csv_path), f"CSV file was not found at {self.csv_path}"
        
        # Store CSV as Dataframe
        self.csv_df = pd.read_csv(self.csv_path)
        
        # Ensure data path validity
        self.img_base_path = root
        assert os.path.isdir(self.img_base_path), f"Need valid data path as `root`. Got {root}"
        self.img_concrete_path = os.path.join(
            self.img_base_path,
            mode 
        )
        assert os.path.isdir(self.img_concrete_path), f"Could not find valid data path at {self.img_concrete_path}"
        
        self.img_frmt_ext = img_ext
        self.transform = None 
        self.target_transform = None
        
        self.num_classes = len(self.class_id_map)
        self.class_names = list(self.class_id_map.keys())
        # All `target` values of the dataset
        self.labels = list(map(
            lambda idx: self.get_sparse_label(self.csv_df, idx),
            range(len(self.csv_df))
        ))

        print(f"Getting '{mode}' data from {self.img_concrete_path}")


    def __len__(self):
        return len(self.csv_df)


    def __getitem__(self, idx):

        """
        Generates a single (img, label) pair

        NOTE: Shuffling is taken care of by the DataLoader wrapper!
        """
        
        if not isinstance(idx, int):
            idx = idx.item()
        
        img_path = os.path.join(
            self.img_concrete_path, 
            "{0}.{1}".format(
                self.csv_df.iloc[idx, 0], 
                self.img_frmt_ext
            )
        )
        img_label = self.get_sparse_label(self.csv_df, idx)
        
        # read data
        try:
            img_data = io.read_image(img_path).float()
        except Exception as e:
            print("Error when trying to read data file:", e)
            return None 

        # apply transforms
        if self.transform is not None:
            img_data = self.transform(img_data)
        if self.target_transform is not None:
            img_label = self.target_transform(img_label)
        
        # return data, label
        return img_data, img_label


    @staticmethod 
    def return_tensor(func):
        
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            return torch.LongTensor([result])

        return wrapped

    
    @staticmethod
    @return_tensor.__func__
    def get_sparse_label(csv_df, idx):
        
        """ 
        Convert one-hot-encoded labels (from across the Dataframe columns)
        to a single sparse label format
        """

        one_hot_label = csv_df.iloc[idx, 1:]
        for classlabel, value in one_hot_label.items():
            if value == 1:
                return ISIC18_T3_Dataset.class_id_map.get(classlabel)
        return -1
    
