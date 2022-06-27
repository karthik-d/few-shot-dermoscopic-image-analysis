# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    
    """
    PrototypicalBatchSampler: yields a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples' from the configuration,
    
    At every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    
    NOTE: __len__ returns the number of episodes per epoch
    """

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        
        """
        Initialize the PrototypicalBatchSampler object
        
        Args:
        - labels: an iterable containing all the labels for the current dataset
                    samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        """

        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # - Create a matrix, indexes, of dim: classes -by- max-elems-per-class
        # - Fill it with NaNs
        self.idxs = range(len(self.labels))
        self.indexes = torch.Tensor(
            np.empty(
                (len(self.classes), max(self.counts)), 
                dtype=int
            ) * np.nan
        )
        self.numel_per_class = torch.zeros_like(self.classes)
        
        # - For every class c, fill the corresponding row with the indices samples belonging to c
        #   `numel_per_class` stores the number of samples for each class/row
        for idx, label in enumerate(self.labels):
            
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[
                label_idx, 
                np.where(
                    np.isnan(self.indexes[label_idx])
                )[0][0]
            ] = idx

            # update num_elem_per_class
            self.numel_per_class[label_idx] += 1
            

    def __iter__(self):
        
        """
        Generate a batch of indices
        """

        spc = self.sample_per_class
        cpi = self.classes_per_it

        for _ in range(self.iterations):
            
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            
            # Get randomly sampled indices for classes
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                
                s = slice(i * spc, (i + 1) * spc)
                label_idx = torch.arange(
                    len(self.classes)
                ).long()[self.classes == c].item()
                
                # Get randomly sampled indices for samples
                sample_idxs = torch.randperm(
                    self.numel_per_class[label_idx]
                )[:spc]

                batch[s] = self.indexes[label_idx][sample_idxs]

            # Construct batch
            batch = batch[torch.randperm(len(batch))]
            yield batch

    
    def __len__(self):
        
        """
        Returns the number of iterations (episodes) per epoch
        """
        return self.iterations
