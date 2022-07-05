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
        ) 
        # isic18_t3_root_path = "/home/miruna/Skin-FSL/repo/Experiments/data/datasets/ISIC18-T3/ds_phase_1"
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