import os
import argparse

from utils.DotDict import DotDict
from config import config as root_config

config = DotDict()
config.update(root_config)

config.update(
    DotDict(
        logs_path = os.path.join(
            config.logs_root_path,
            'classifier_nw'
        )
    )
)

config.update(
    DotDict(
        epochs = 100,
        learning_rate = 1e-03,
        lr_scheduler_step = 20,
        lr_scheduler_gamma = 0.5,
        iterations = 50,
        classes_per_it_tr = 4,
        num_support_tr = 15,  # 15-shot training
        num_query_tr = 10,    
        classes_per_it_val = 3,
        num_support_val = 3,  # 3-shot testing
        num_query_val = 10,
        classes_per_it_test = 2,
        num_support_test = 3,   
        manual_seed = 7,
        cuda = True
    )
)

config.update(
    DotDict(
        input_channels = 3,
        input_xy = (224, 224)
    )
)

config.update(
    DotDict(
        input_shape = (config.input_channels, *config.input_xy)
    )
)


config.update(
    DotDict(
        nonmeta_batchsize_tr = 64,
        nonmeta_batchsize_val = 64
    )
)


config.update(
    DotDict(
        classifier_name = 'L_SVM'
    )
)