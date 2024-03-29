from architectures.metaderm import MetaDerm
from architectures.protonet import ProtoNet
from architectures.metaderm_lr import MetaDerm_LR
from architectures.resnet18_lr import ResNet18_LR
from architectures.resnet50_lr import ResNet50_LR
from .prototypical_batch_sampler import PrototypicalBatchSampler
from .crossentropy_loss import get_crossentropy_loss_fn
from . import transforms
#from omniglot_dataset import OmniglotDataset

from .config import config
from data.config import config as data_config
from data.Derm7Pt_Dataset import Derm7Pt_Dataset
from utils import helpers

from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plot
import numpy as np
import torch
import os


# TODO: Parameterize num_classes and class selection

def init_seed(config):
    torch.cuda.cudnn_enabled = True
    np.random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed(config.manual_seed)


def init_dataset(config, data_config, mode):

    if mode in ['train', 'val']:
        allowed_labels = Derm7Pt_Dataset.get_class_ids(
            data_config.derm7pt_train_classes
        )
    else:
        allowed_labels = Derm7Pt_Dataset.get_class_ids(
            data_config.derm7pt_test_classes
        )

    dataset = Derm7Pt_Dataset(
        mode=mode, 
        root=data_config.derm7pt_root_path,
        transform=transforms.compose_transforms([
            transforms.get_resize_transform()
        ]),
        allowed_labels=allowed_labels
    )

    print(len(dataset))

    # Ensure classes count
    if dataset.num_classes < config.classes_per_it_tr or dataset.num_classes < config.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))

    return dataset


def init_sampler(config, labels, mode):

    if mode == 'train':
        classes_per_it = config.classes_per_it_tr
        num_support = config.num_support_tr 
        num_query = config.num_query_tr
    else:
        classes_per_it = config.classes_per_it_val
        num_support = config.num_support_val 
        num_query = config.num_query_val

    # Initialize and return the batch sampler
    return PrototypicalBatchSampler(
        labels=labels,
        classes_per_it=classes_per_it,
        num_support=num_support,
        num_query=num_query,
        iterations=config.iterations
    )


def init_dataloader(config, data_config, mode):

    # Make dataset and sampler
    dataset = init_dataset(config, data_config, mode)
    sampler = init_sampler(config, dataset.labels, mode)
    
    # Wrap the dataset into torch's dataloader
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler
    ), sampler


def init_dataloader_nonmeta(config, data_config, mode):

    # Make dataset 
    dataset = init_dataset(config, data_config, mode)

    if mode == 'train':
        batch_size = config.nonmeta_batchsize_tr
    elif mode == 'val':
        batch_size = config.nonmeta_batchsize_val

    # Wrap dataset into dataloader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


def init_loss_fn(data_config, sampler, mode):
    
    # bind sampler and return loss function
    if mode in ['train', 'val']:
        class_names = data_config.derm7pt_train_classes 
    else:
        class_names = data_config.derm7pt_test_classes

    return get_crossentropy_loss_fn(
        classes=Derm7Pt_Dataset.get_class_ids(class_names),
        sampler=sampler
    )


def init_loss_fn_nonmeta(mode):

    if mode == 'train':
        class_names = data_config.derm7pt_train_classes 
    else:
        class_names = data_config.derm7pt_test_classes
    
    print(Derm7Pt_Dataset.class_id_map)
    print(Derm7Pt_Dataset.get_class_ids(class_names))
    return get_crossentropy_loss_fn(
        classes=Derm7Pt_Dataset.get_class_ids(class_names),
        sampler=None
    )


def init_protonet(config):
    
    """
    Initialize the ProtoNet architecture
    """

    device = 'cuda:0' if (torch.cuda.is_available() and config.cuda) else 'cpu'
    return ProtoNet().to(device)


def init_metaderm(config, data_config):
    
    """
    Initialize the MetaDerm architecture
    """

    device = 'cuda:0' if (torch.cuda.is_available() and config.cuda) else 'cpu'
    model = ResNet50_LR(
        num_classes=len(data_config.derm7pt_train_classes)
    ).to(device)

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


def run_concrete_train_loop(
    config, 
    tr_dataloader,
    tr_loss_fn, 
    model, 
    optim, 
    lr_scheduler, 
    val_dataloader=None,
    val_loss_fn=None
):
    
    """ 
    Run the concrete training loop on the model 
    with the prototypical learning algorithm
    """

    device = 'cuda:0' if (torch.cuda.is_available() and config.cuda) else 'cpu'

    if val_dataloader is None:
        best_state = None

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(
        config.logs_path, 
        'best_model.pth'
    )
    last_model_path = os.path.join(
        config.logs_path, 
        'last_model.pth'
    )

    # Delete existing log file
    with open('train_log.txt', 'w'):
        pass

    for epoch in range(config.epochs):
        print(f'=== Epoch: {epoch} ===')

        tr_iter = iter(tr_dataloader)
        model.train()

        # TRAINING STEP --
        for batch in tqdm(tr_iter):
            
            # retrace gradient and propagate batch
            optim.zero_grad()            
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
            loss, acc = tr_loss_fn(
                model_output, 
                target=y,
                get_prediction_results=False
            )
            
            loss.backward()
            optim.step()

            train_loss.append(loss.item())
            train_acc.append(acc.item())

        # Compute training stats
        avg_loss = np.mean(train_loss[-config.iterations:])
        avg_acc = np.mean(train_acc[-config.iterations:])

        print(f'Avg Train Loss: {avg_loss}, Avg Train Acc: {avg_acc}')
        lr_scheduler.step()

        if val_dataloader is None or val_loss_fn is None:
            continue

        val_iter = iter(val_dataloader)
        model.eval()

        # VALIDATION STEP --
        for batch in tqdm(val_iter):
            
            # only propagate batch
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
            loss, acc = tr_loss_fn(
                model_output, 
                target=y,
                get_prediction_results=False
            )
                
            val_loss.append(loss.item())
            val_acc.append(acc.item())

        # Compute validation stats
        avg_loss_val = np.mean(val_loss[-config.iterations:])
        avg_acc_val = np.mean(val_acc[-config.iterations:])

        
        # Save best model --> replaced if it beats current best
        if avg_acc_val >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc_val
            best_state = model.state_dict()

        # Save current model --> replaced at each epoch
        torch.save(model.state_dict(), last_model_path)
        print(f'Avg Val Loss: {avg_loss_val}, Avg Val Acc: {avg_acc_val}, Best Acc (train): {best_acc}')

        # LOG training stats
        helpers.save_list_to_file(
            os.path.join(
                config.logs_path,
                'train_log.txt'
            ), 
            [
                value  
                for value in [epoch, avg_loss, avg_acc, avg_loss_val, avg_acc_val]
            ]
        )

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


# TODO: Change to `validate` to run only the  validation loop;
# TODO: Plug into the training loop
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


# TODO: Change to `validate` to run only the  validation loop;
# TODO: Plug into the training loop
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
    # model = init_metaderm(config)
    # model_path = os.path.join(
    #     config.logs_path, 
    #     'best_model.pth'
    # )
    # model.load_state_dict(torch.load(model_path))

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

    tr_dataloader = init_dataloader_nonmeta(
        config=config, 
        data_config=data_config,
        mode='train'
    )
    val_dataloader = init_dataloader_nonmeta(
        config=config, 
        data_config=data_config,
        mode='val'
    )

    tr_loss_fn = init_loss_fn_nonmeta(mode='train')
    val_loss_fn = init_loss_fn_nonmeta(mode='val')

    model = init_metaderm(config, data_config)

    # Continue training
    # model_path = os.path.join(
    #     config.logs_path, 
    #     'best_model.pth'
    # )
    # model.load_state_dict(torch.load(model_path), strict=False)

    optim = init_optim(config, model)
    lr_scheduler = init_lr_scheduler(config, optim)
    train_stats = run_concrete_train_loop(
        config=config,
        tr_dataloader=tr_dataloader,
        val_dataloader=val_dataloader,
        tr_loss_fn=tr_loss_fn,
        val_loss_fn=val_loss_fn,
        model=model,
        optim=optim,
        lr_scheduler=lr_scheduler
    )
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = train_stats


if __name__ == '__main__':
    # main()
    pass
