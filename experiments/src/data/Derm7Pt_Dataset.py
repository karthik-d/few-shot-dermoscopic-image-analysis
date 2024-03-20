from torch.utils.data import Dataset
from torchvision import io 
import os 
import pandas as pd
import torch

from .config import config
from .derm7pt.dataset import Derm7PtDataset, Derm7PtDatasetGroupInfrequent

class Derm7Pt_Dataset(Dataset):

    """
    Extended custom `Dataset` class to interface
    torch with the Derm7Pt T3 Dataset
    """

    class_id_map = dict(
        BCC = 0,
        NEV = 1,
        MEL = 2,
        MISC = 3,
        SK = 4,
    )


    def __init__(self, root, mode='train', img_type='derm', allowed_labels=None, transform=None, target_transform=None):
        
        """
        Initialize the Dataset
        - root: Root directory containing all data files
        - mode: Determines the CSV file to use (ISIC18_T3_<mode>.csv)
                Also determined the subdirectory for data split
                Valid values: <subdirectory names within the data root>

        NOTE: CSV must be present in the [ROOT]/data directory
        """
        
        super(Derm7Pt_Dataset, self).__init__()

        dir_release = root #'/home/miruna/Skin-FSL/Derm7pt/release_v0'
        dir_meta = os.path.join(dir_release, 'meta')
        dir_images = os.path.join(dir_release, 'images')

        meta_df = pd.read_csv(os.path.join(dir_meta, 'meta.csv'))
        train_indexes = list(pd.read_csv(os.path.join(dir_meta, 'train_indexes.csv'))['indexes'])
        valid_indexes = list(pd.read_csv(os.path.join(dir_meta, 'valid_indexes.csv'))['indexes'])
        test_indexes = list(pd.read_csv(os.path.join(dir_meta, 'test_indexes.csv'))['indexes'])

        derm_data = Derm7PtDataset(dir_images=dir_images, 
                        metadata_df=meta_df.copy(), # Copy as is modified.
                        train_indexes=train_indexes, valid_indexes=valid_indexes, 
                        test_indexes=test_indexes)

        derm_data_group = Derm7PtDatasetGroupInfrequent(dir_images=dir_images, 
                                                    metadata_df=meta_df.copy(), # Copy as is modified.
                                                    train_indexes=train_indexes, 
                                                    valid_indexes=valid_indexes, 
                                                    test_indexes=test_indexes)

        # Derm7Pt_Dataset.class_id_map = {
        #     label: idx 
        #     for idx, label in enumerate(derm_data_group.get_label_by_abbrev('DIAG').abbrevs.values)
        # }

        if mode == 'val':
            mode = 'valid'
        self.mode = mode
        
        _path_list = derm_data_group.get_img_paths(data_type=mode, img_type=img_type)
        _labels_df = derm_data_group.get_labels(data_type=mode, one_hot=False)['DIAG']
        self.path_list = []
        self.labels = []
        for label, path in zip(_labels_df, _path_list):
            if label in allowed_labels:
                self.path_list.append(path)
                self.labels.append(label)
        
        # Ensure data path validity
        self.img_base_path = root
        assert os.path.isdir(self.img_base_path), f"Need valid data path as `root`. Got {root}"
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.num_classes = len(self.class_id_map)
        self.class_names = list(self.class_id_map.keys())
        # All `target` values of the dataset      

        print(f"Getting '{mode}' data from {self.img_base_path} and {allowed_labels}")


    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, idx):

        """
        Generates a single (img, label) pair

        NOTE: Shuffling is taken care of by the DataLoader wrapper!
        """
        
        if not isinstance(idx, int):
            idx = idx.item()
        
        img_path = os.path.join(
            self.img_base_path, 
            self.path_list[idx]
        )
        img_label = self.labels[idx]
        
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
    def get_class_ids(class_names):
        return [
            Derm7Pt_Dataset.class_id_map.get(x)
            for x in class_names
        ]


    @staticmethod 
    def return_tensor(func):
        
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            return torch.Tensor(result)

        return wrapped

    
    
    
