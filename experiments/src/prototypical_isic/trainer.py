from architectures.metaderm import MetaDerm
from architectures.protonet import ProtoNet
from .exhaustive_batch_sampler import ExhaustiveBatchSampler
from .prototypical_loss import prototypical_loss as loss_fn
from . import transforms
#from omniglot_dataset import OmniglotDataset

from prototypical.config import config
from data.config import config as data_config
from data.ISIC18_T3_Dataset import ISIC18_T3_Dataset
from utils import helpers

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


def init_sampler(config, class_names, labels, mode):

    if mode == 'train':
        classes_per_it = config.classes_per_it_tr
        num_samples = config.num_support_tr + config.num_query_tr
    else:
        classes_per_it = config.classes_per_it_val
        num_samples = config.num_support_val + config.num_query_val

    # Initialize and return the batch sampler
    return ExhaustiveBatchSampler(
        class_names=class_names,
        labels=labels,
        classes_per_it=classes_per_it,
        num_samples=num_samples,
        iterations=config.iterations
    )


def init_dataloader(config, data_config, mode):

    # Make dataset and samples
    dataset = init_dataset(config, data_config, mode)
    sampler = init_sampler(config, dataset.class_names, dataset.labels, mode)
    
    # Wrap the dataset into torch's dataloader
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler
    )


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
    model = MetaDerm().to(device)
    print(model)
    return model


def init_optim(config, model):

    return torch.optim.Adam(
        params=model.parameters(),
        lr=config.learning_rate
    )


def init_lr_scheduler(config, optim):

    return torch.optim.lr_scheduler.StepLR(
        optimizer=optim,
        gamma=config.lr_scheduler_gamma,
        step_size=config.lr_scheduler_step
    )


def run_concrete_test_loop(config, test_dataloader, model):
    
    """ 
    Run a trained model through the test dataset
    """

    device = 'cuda:0' if (torch.cuda.is_available() and config.cuda) else 'cpu'
    avg_acc = []

    # Test as average of 5 iterations
    for epoch in range(5):

        test_iter = iter(test_dataloader)
        for batch in tqdm(test_iter):
            
            x, y = batch
            x, y = x.to(device), y.to(device)

            model_output = model(x)
            _, acc = loss_fn(
                model_output, 
                target=y,
                n_support=config.num_support_val
            )
            avg_acc.append(acc.item())
    
    # Compute average stats
    avg_acc = np.mean(avg_acc)
    print(f'Test Acc: {avg_acc}')

    return avg_acc


def test():
    
    """
    Initialize all parameters and test the model
    - driver wrapper for model testing
    """

    if torch.cuda.is_available() and not config.cuda:
        print("CUDA device available and unused")

    # load dataset
    init_seed(config)
    test_dataloader = init_dataloader(
        config=config, 
        data_config=data_config, 
        mode='test'
    )

    # load model
    model = init_metaderm(config)
    model_path = os.path.join(
        config.logs_path, 
        'best_model.pth'
    )
    model.load_state_dict(torch.load(model_path))

    # run test
    run_concrete_test_loop(
        config=config,
        test_dataloader=test_dataloader,
        model=model
    )


def train():
    
    """
    Initialize all parameters and train the model
    - driver wrapper for model training
    """
    
    if not os.path.exists(config.logs_path):
        os.makedirs(config.logs_path)

    if torch.cuda.is_available() and not config.cuda:
        print("CUDA device available and unused")

    init_seed(config)

    tr_dataloader = init_dataloader(
        config=config, 
        data_config=data_config,
        mode='train'
    )
    val_dataloader = init_dataloader(
        config=config, 
        data_config=data_config,
        mode='val'
    )

    model = init_metaderm(config)
    optim = init_optim(config, model)
    lr_scheduler = init_lr_scheduler(config, optim)
    train_stats = run_concrete_train_loop(
        config=config,
        tr_dataloader=tr_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optim=optim,
        lr_scheduler=lr_scheduler
    )
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = train_stats


def run():

    training_data = ISIC18_T3_Dataset(
	    os.path.join(data_config.csv_path, data_config.isic18_t3_train_csv),
	    os.path.join(data_config.data_path, data_config.isic18_t3_train_dir)
	)

    train_dataloader = DataLoader(
        training_data, 
        batch_size=8, 
        shuffle=True
    )

    imgs, labels = next(iter(train_dataloader))
    print(labels[0].squeeze())
    plot.imshow(imgs[0])
    plot.show()

    print("Works!")

if __name__ == '__main__':
    # main()
    pass
