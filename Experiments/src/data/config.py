from utils.DotDict import DotDict

import os

config = DotDict(
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *(os.path.pardir,)*2, 'data'),
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *(os.path.pardir,)*2, 'data', 'datasets'),

    isic18_t3_train_csv = 'ISIC18_T3_Train.csv',
    isic18_t3_val_csv = 'ISIC18_T3_Validation.csv',
    isic18_t3_root_dir = 'ISIC18-T3',
    isic18_t3_train_dir = os.path.join('ISIC18-T3', 'train'),
    isic18_t3_val_dir = os.path.join('ISIC18-T3', 'val')
)   
