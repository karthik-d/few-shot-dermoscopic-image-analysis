from architectures.metaderm_lr import MetaDerm_LR
from architectures.protonet import ProtoNet
from .exhaustive_extended_batch_sampler import ExhaustiveExtendedBatchSampler
from .prototypical_loss import get_prototypical_loss_fn
from . import transforms

from prototypical.config import config
from data.config import config as data_config
from data.ISIC18_T3_Dataset import ISIC18_T3_Dataset
from utils import helpers, displayers

from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plot
import numpy as np
import torch
import os


def init_seed(config):
    torch.cuda.cudnn_enabled = True
    np.random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed(config.manual_seed)


def init_dataset(config, data_config, mode):

    dataset = ISIC18_T3_Dataset(
        mode=mode, 
        root=data_config.isic18_t3_root_path,
        transform=transforms.compose_transforms([
            transforms.get_resize_transform()
        ])
    )

    # Ensure classes count
    if dataset.num_classes < config.classes_per_it_tr or dataset.num_classes < config.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))

    return dataset


def init_sampler(config, data_config, labels, mode):

    if mode == 'train':
        classes_per_it = config.classes_per_it_tr
        num_samples = config.num_support_tr + config.num_query_tr
    else:
        classes_per_it = config.classes_per_it_val
        num_samples = config.num_support_val + config.num_query_val

    # Initialize and return the batch sampler 
    # exhaustively makes 1 query per iteration
    return ExhaustiveExtendedBatchSampler(
        class_names=class_names,
        support_class_names=data_config.test_classes,
        query_class_names=data_config.test_classes,
        labels=labels,
        classes_per_it=classes_per_it,
        num_support=config.num_support_test,
        force_support=[]    
    )


def init_dataloader(config, data_config, mode):

    # Make dataset and samples
    dataset = init_dataset(config, data_config, mode)
    sampler = init_sampler(config, data_config, dataset.labels, mode)
    
    # Wrap the dataset into torch's dataloader
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler
    ), dataset, sampler


def init_loss_fn(sampler):
    
    # bind sampler and return loss function
    return get_prototypical_loss_fn(sampler=sampler)


def init_protonet(config):
    
    """
    Initialize the ProtoNet architecture
    """

    device = 'cuda:0' if (torch.cuda.is_available() and config.cuda) else 'cpu'
    return ProtoNet().to(device)


def init_metaderm(config):
    
    """
    Initialize the MetaDerm architecture
    """

    device = 'cuda:0' if (torch.cuda.is_available() and config.cuda) else 'cpu'
    model = MetaDerm_LR(num_classes=None).to(device)
    print(model)
    return model


def compute_accuracy(labels, predictions):
    pass


def run_concrete_test_loop(config, data_config, test_dataloader, loss_fn, model, dataset):
    
    """ 
    Run a trained model through the test dataset
    """

    device = 'cuda:0' if (torch.cuda.is_available() and config.cuda) else 'cpu'
    avg_acc = []

    # Test as average of 5 iterations
    for epoch in range(5):

        all_predictions = torch.tensor([], dtype=torch.long)
        all_truths = torch.tensor([], dtype=torch.long)
        test_iter = iter(test_dataloader)
        for batch in tqdm(test_iter):
            x, y = batch
            
            # Pack into batch (for single instance)
            if len(x.shape)==3:
                x = x.unsqueeze(0)

            # Decode data from tensor
            if isinstance(x, torch.Tensor):
                x = x.to(device)
            if isinstance(y, torch.Tensor):
                y = y.to(device)      
            
            model_output = model(x)
            acc = local_classifier(
                model_output
            )

            avg_acc.append(acc.item())

            # gather predictions
            all_predictions = torch.cat([
                all_predictions,
                predictions
            ])

            # gather truths
            all_truths = torch.cat([
                all_truths,
                truths
            ])

        confusion_matrix = displayers.get_printable_confusion_matrix(
            all_labels=all_truths,
            all_predictions=all_predictions,
            classes=data_config.test_classes
        )
        print("\nClassification Confusion Matrix\n")
        print(confusion_matrix)

        avg_acc_val = np.mean(avg_acc)
        print(f'\nAverage Test Acc: {avg_acc_val}')
    
    # Compute average stats
    avg_acc_val = np.mean(avg_acc)
    print(f'\nAverage Test Acc: {avg_acc_val}')

    return avg_acc


# TODO: Produce DOCKER file for submission to ISIC Challenge Website
def test():
    
    """
    Initialize all parameters and test the model
    - driver wrapper for model testing
    """

    if torch.cuda.is_available() and not config.cuda:
        print("CUDA device available and unused")

    # load dataset
    init_seed(config)
    test_dataloader, test_dataset, sampler = init_dataloader(
        config=config, 
        data_config=data_config, 
        mode='test'
    )

    loss_fn = get_prototypical_loss_fn(sampler)

    # load model
    model = init_metaderm(config)
    # model_path = os.path.join(
    #     config.logs_path, 
    #     'best_model.pth'
    # )

    model_path = os.path.join(
        # '/home/miruna/Skin-FSL/repo/Experiments/data/datasets/ISIC18-T3/ds_phase_1',
        config.logs_path,
        'best_model.pth'
    )
    model.load_state_dict(torch.load(model_path))

    # run test
    run_concrete_test_loop(
        config=config,
        data_config=data_config,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        model=model,
        dataset=test_dataset
    )

