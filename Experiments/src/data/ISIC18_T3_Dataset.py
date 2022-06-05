from torch.utils.data import Dataset
from torchvision import io 
import os 
import pandas as pd

class ISIC18_T3_Dataset(Dataset):

    class_id_map = dict(
        MEL = 0,
        NV = 1,
        BCC = 2,
        AKIEC = 3,
        BKL = 4,
        DF = 5,
        VASC = 6
    )

    def __init__(self, csv_path, data_path, img_ext='jpg', transform=None, target_transform=None):
        self.csv_df = pd.read_csv(csv_path)
        self.img_base_path = data_path
        self.img_frmt_ext = img_ext
        self.transform = None 
        self.target_transform = None

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        # shuffling is taken care of by the DataLoader wrapper!
        img_path = os.path.join(self.img_base_path, self.csv_df.iloc[idx, 0]+self.img_frmt_ext)
        img_label = self.get_sparse_label(self.csv_df, self.idx)
        # read and transform data
        img_data = io.read_image(img_path)
        if self.transform:
            img_data = self.transform(img_data)
        if self.target_transform:
            img_label = self.target_transform(img_label)
        # return data, label
        return img_data, img_label

    @staticmethod
    def get_sparse_label(csv_df, idx):
        one_hot_label = csv_df.iloc[idx, 1:]
        for index, value in one_hot_label.items():
            if value == 1:
                return ISIC_T3_Dataset.class_id_map.get(index)
        return -1
