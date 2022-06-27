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
        )
    )
)