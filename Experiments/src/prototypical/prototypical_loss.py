import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return self.prototypical_loss(input, target, self.n_support)

    
    @staticmethod
    def euclidean_dist(x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)


def get_prototypical_loss_fn(sampler):

    def compute_loss(input, target):

        """
        Compute the barycentres by averaging the features of n_support
        samples for each class in target, computes then the distances from each
        samples' features to each one of the barycentres, computes the
        log_probability for each n_query samples for each one of the current
        classes, of appartaining to a class c, loss and accuracy are then computed
        and returned
        Args:
        - input: the model output for a batch of samples
        - target: ground truth for the above batch of samples
        - n_support: number of samples to keep in account when computing
        barycentres, for each one of the current classes
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
        query_classes = torch.unique(target_cpu[query_idxs])
        n_query_classes = query_classes.size(0)

        dists = PrototypicalLoss.euclidean_dist(query_samples, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

        target_inds = torch.tensor(list(range(len(query_classes))))
        target_inds = target_inds.view(n_query_classes, 1, 1)
        target_inds = target_inds.expand(n_query_classes, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

        return loss_val,  acc_val

    # Return wrapped loss function, with sampler bound
    return compute_loss

        


