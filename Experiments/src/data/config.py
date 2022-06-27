from utils.DotDict import DotDict

import os

config = DotDict()

config.update(
    DotDict(
        root_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            *(os.path.pardir,)*2
        ) 
    )
)

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
        isic18_t3_root_path = os.path.join(config.data_root_path, 'ISIC18-T3') 
    )
)

config.update(
    DotDict(               
        isic18_t3_train_path = os.path.join(config.isic18_t3_root_path, 'train'),
        isic18_t3_val_path = os.path.join(config.isic18_t3_root_path, 'val')
    )
)