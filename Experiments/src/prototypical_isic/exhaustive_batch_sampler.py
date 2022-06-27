import numpy as np
import torch


class ExhaustiveBatchSampler(object):
    
    """
    ExhaustiveBatchSampler: yields a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples' from the configuration,
    
    At every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    
    NOTE: __len__ returns the number of episodes per epoch
    """

    def __init__(self, class_names, labels, classes_per_it, num_samples, iterations):
        
        """
        Initialize the object
        
        Args:
        - labels: an iterable containing all the labels for the current dataset
                    samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        - class_names: ordered list of class names
        """

        super(ExhaustiveBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.class_names = class_names 
        # Stores all class indices in order (0, 1, .., n)
        self.classes = torch.LongTensor(range(len(self.class_names)))
        self.counts = [ 
            np.count_nonzero(self.labels==class_idx) 
            for class_idx in len(self.classes) 
        ] 

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

        NOTE: Takes the first `cpi` number of classes -- cpi is just a convenience param, 
                semantically, all classes need to be sampled in the Exhaustive Batch Sampler
        """

        spc = self.sample_per_class
        cpi = self.classes_per_it

        for _ in range(self.iterations):
            
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            
            # Obtain all class indices (in order)
            for label_idx, c in enumerate(self.classes):
                
                s = slice(i * spc, (i + 1) * spc)

                # Randomly sample samples for the class
                # Replicate sampling if class does NOT have sufficient samples
                sample_idxs = torch.empty(0, dtype=torch.int8)
                while(len(sample_idxs)!=spc):
                    remain = spc - len(sample_idxs)
                    sample_idxs = torch.cat([
                        sample_idxs,
                        torch.randperm(
                            self.numel_per_class[label_idx]
                        )[:min(remain, spc)]
                    ])
                batch[s] = self.indexes[label_idx][sample_idxs]

            # Construct batch - shuffle the generated batches
            batch = batch[torch.randperm(len(batch))]
            yield batch

    
    def __len__(self):
        
        """
        Returns the number of iterations (episodes) per epoch
        """
        return self.iterations
