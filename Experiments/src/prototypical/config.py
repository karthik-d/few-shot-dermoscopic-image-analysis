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
            'prototypical'
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
        classes_per_it_val = 2,
        num_support_val = 3,  # 3-shot testing
        num_query_val = 10,
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

# Descriptions

"""
parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='..' + os.sep + 'dataset')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='..' + os.sep + 'output')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=60)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=5)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=15)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

"""
