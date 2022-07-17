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


def get_crossentropy_loss_fn(sampler=None):

    loss_fn = nn.CrossEntropyLoss()

    def compute_loss(input, target, get_prediction_results=False):

        """
        Args:
        - input: the model output for a batch of samples
        - target: ground truth for the above batch of samples
        - get_prediction_results: If True, a third return value corresponds to 
                                    (predicted_labels, actual_labels) 
        """

        target_cpu = target.to('cpu')
        input_cpu = input.to('cpu')

        classes = torch.unique(target_cpu)
        n_classes = len(classes)

        (support_idxs, query_idxs) = sampler.decode_batch(
            batch_labels=target_cpu, 
            batch_classes=classes
        )
        n_support = len(support_idxs[0])
        n_query = len(query_idxs[0])

        prototypes = torch.stack([
            input_cpu[idx_list].mean(0) 
            for idx_list in support_idxs
        ])

        query_idxs = torch.stack(query_idxs).view(-1).long()
        query_samples = input_cpu[query_idxs]
        true_query_classes = target_cpu[query_idxs]
        n_query_classes = torch.unique(true_query_classes).size(0)

        # prototypes and dists --> class ordering is same as that of `classes`
        dists = CrossEntropyLoss.euclidean_dist(query_samples, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_query_classes, n_query, -1)

        # target_inds is specific to gathering labels from `log_p_y`
        # indexes correspond to `true_query_classes`
        target_inds = torch.tensor(list(range(n_query_classes)))
        target_inds = target_inds.view(n_query_classes, 1, 1)
        target_inds = target_inds.expand(n_query_classes, n_query, 1).long()

        # The relative ordering of `target_inds` and `log_p_y` classes must be consistent
        # i.e. log_p_y has the clases in same order in both 2nd and 3rd dimensions
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        # Predicted indices are based on the `classes` var (obviously!)
        pred_query_classes = classes[y_hat.flatten()]
        acc_val = true_query_classes.eq(pred_query_classes).float().mean()    

        if not get_prediction_results:
            return loss_val,  acc_val
        else:
            return loss_val, acc_val, (pred_query_classes, true_query_classes)

    # Return wrapped loss function, with sampler bound (if applicable)
    if sampler is None:
        loss_mode = 'nonmeta'
        return loss_fn
    else: 
        loss_mode = 'meta'
        return compute_loss

    

        


