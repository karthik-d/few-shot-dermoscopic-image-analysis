from utils.DotDict import DotDict
from config import config as root_config

import os

config = DotDict()
config.update(root_config)

config.update(
    DotDict(
        csv_root_path = os.path.join(
            config.root_path,
            'data'
        ),
        # csv_root_path = "/home/miruna/Skin-FSL/repo/Experiments/data/datasets/ISIC18-T3/ds_phase_1/",
        data_root_path = os.path.join(
            config.root_path, 
            'data', 
            'datasets'
        ),
    )
)

config.update(
    DotDict(
        isic18_t3_root_path = os.path.join(
            config.data_root_path, 
            'ISIC18-T3'
        ),
        derm7pt_root_path = '/home/miruna/Skin-FSL/Derm7pt/release_v0',
		ph2_root_path = os.path.join(
            config.data_root_path, 
            'PH2_Dataset'
        )
    )
)

config.update(
    DotDict(               
        isic18_t3_train_path = os.path.join(
            config.isic18_t3_root_path, 
            'train'
        ),
        isic18_t3_val_path = os.path.join(
            config.isic18_t3_root_path, 
            'val'
        ),
        isic18_t3_test_path = os.path.join(
            config.isic18_t3_root_path, 
            'test'
        )
    )
)

config.update(
    DotDict(
        test_classes = [
            'AKIEC',
            'VASC',
            'DF'
        ],
        train_classes = [
            'MEL',
            'NV',
            'BCC',
            'BKL'
        ],
        derm7pt_test_classes = [
            'BCC',
            'SK',
            'MISC'
        ],
        derm7pt_train_classes = [
            'NEV',
            'MEL'
        ]
    )
)