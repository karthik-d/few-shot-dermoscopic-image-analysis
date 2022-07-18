import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import torch.nn as nn


class CrossEntropyLoss(Module):

    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self, n_support):
        super(CrossEntropyLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return self.crossentropy_loss(input, target, self.n_support)

    
    @staticmethod
    def euclidean_dist(x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)


def get_crossentropy_loss_fn(classes, sampler=None):

    loss_fn = nn.CrossEntropyLoss()

    def compute_loss_nonmeta(input, target, get_prediction_results=False):

        """
        Args:
        - input: the model output for a batch of samples
        - target: ground truth for the above batch of samples
        - get_prediction_results: If True, a third return value corresponds to 
                                    (predicted_labels, actual_labels) 
        """

        def accuracy(input_cpu, class_posns_cpu):
            _, pred_idxs = input_cpu.max(1) 
            acc_val = class_posns_cpu.eq(pred_idxs).float().mean() 
            return acc_val

        input_cpu = input.to('cpu')
        print(classes)
        print(target)
        class_posns_cpu = torch.LongTensor(
            [
                classes.index(x)
                for x in target
            ]
        ).to('cpu')

        return (
            loss_fn(input_cpu, class_posns_cpu), 
            accuracy(
                input_cpu, 
                class_posns_cpu.unsqueeze(0)
            )
        )


    # Return wrapped loss function, with sampler bound (if applicable)
    if sampler is None:
        loss_mode = 'nonmeta'
        return compute_loss_nonmeta
    # else: 
    #     loss_mode = 'meta'
    #     return compute_loss_meta

    

        


