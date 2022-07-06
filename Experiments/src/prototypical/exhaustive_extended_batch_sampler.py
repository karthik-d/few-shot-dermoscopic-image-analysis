import numpy as np
import torch


class ExhaustiveExtendedBatchSampler(object):
    
    """
    NOTE: 

    ExhaustiveExtendedBatchSampler: yields a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_support' from the configuration,

    - Selects a single query at random
    - Samples `classes_per_it` classes randomly, such that the class of the query is included
    - Selects a support set with `num_support` instances from each of those classes
    
    At every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    
    NOTE: __len__ returns the number of data_pts contained in the dataset
    """

    def __init__(self, class_names, labels, support_class_names, query_class_names, classes_per_it, num_support, force_support=[]):
        
        """
        Initialize the object
        
        Args:
        - labels: an iterable containing all the labels for the current dataset
                    samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_support: number of samples for each iteration for each class
        - class_names: ordered list of class names
        - force_support: an iterable of classes to always include in the sampled support 
                        (Requires that size of force_support is lower than `num_support`)
        """
        

        super(ExhaustiveExtendedBatchSampler, self).__init__()
        self.labels = np.array(labels)
        self.classes_per_it = classes_per_it
        self.support_per_class = num_support
        self.iterations = len(self.labels)  # 1-query per iteration, cover all data-pts
        
        self.class_names = class_names 
        # Stores all class indices in order (0, 1, .., n)
        self.classes = torch.LongTensor(range(len(self.class_names)))
        self.support_classes = torch.LongTensor([
            idx for idx in self.classes 
            if self.class_names[idx] in support_class_names
        ])
        self.query_classes = torch.LongTensor([
            idx for idx in self.classes 
            if self.class_names[idx] in query_class_names
        ])

        # Validate force_support count
        assert len(force_support) < self.classes_per_it, "More forced support classes than allowed classes per iteration!"
        assert self.classes_per_it <= len(self.classes), "Insufficient classes for specified `classes_per_it`"

        self.force_support_classes = [
            idx for idx in self.classes 
            if self.class_names[idx] in force_support
        ]

        # Accumulate counts for each class
        self.counts = [ 
            np.count_nonzero(self.labels==class_idx.item()) 
            for class_idx in self.classes
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
        
        # - For every class c, fill the corresponding row with the indices of samples belonging to c
        #   `numel_per_class` stores the number of samples for each class/row
        for idx, label in enumerate(self.labels):
            sample_idx = np.argwhere(self.classes == label).item()
            self.indexes[
                sample_idx, 
                np.where(
                    np.isnan(self.indexes[sample_idx])
                )[0][0]
            ] = idx

            # update num_elem_per_class
            self.numel_per_class[sample_idx] += 1

        print(self.numel_per_class)
        print("Support Domain:", self.support_classes)
        print("Query Domain:", self.query_classes)
            

    def __iter__(self):
        
        """
        Generate a batch of indices
        """

        spc = self.support_per_class   # number of shots
        cpi = self.classes_per_it
        random_cpi = cpi - (1 + self.force_support_classes.size(dim=0))

        # Iterate over `self.indexes` to get each sample
        for query_label in self.query_classes:
            print(query_label, "with", self.numel_per_class[query_label])

            for query_sample_idx in range(self.indexes.size(dim=1)):
                
                query_idx = self.indexes[query_label][query_sample_idx]       
                if torch.isnan(query_idx):
                    # Exhausted all labels in the class
                    break
                else:
                    query_idx = int(query_idx.item())

                # Sample classes for the support set
                # Allow only classes with non-zero samples in them
                # Ensure that `force_support_classes` are always selected
                class_pool = [
                    idx for idx in self.support_classes
                    if (idx!=query_label and self.numel_per_class[idx]!=0)
                ]

                # Count and sample random classes
                num_random_classes = random_cpi 
                fixed_classes = self.force_support_classes[:] + [query_label]
                if query_label in self.force_support_classes:
                    num_random_classes += 1
                    fixed_classes = self.force_support_classes[:]
                c_idxs = np.concatenate(
                    np.random.permutation(class_pool)[:num_random_classes],
                    np.array(fixed_classes)
                )

                # Prepare batch
                batch_size = (spc * cpi) + 1  # (support + 1 query per iteration
                batch = torch.LongTensor(batch_size)
                for i, class_label in enumerate(c_idxs):

                    s = slice(i*spc, (i+1)*spc)                    
                    # Randomly sample samples for the class
                    # Replicate sampling if class does NOT have sufficient samples
                    sample_idxs = torch.empty(0, dtype=torch.long)
                    while(len(sample_idxs)!=spc):
                        
                        remain = spc - len(sample_idxs)
                        curr_sample_idxs = torch.randperm(
                            self.numel_per_class[class_label]
                        )[:min(remain, spc)]

                        # Skip if query is sampled into support
                        if(query_idx in sample_idxs):
                            continue

                        sample_idxs = torch.cat([
                            sample_idxs,
                            curr_sample_idxs
                        ])

                    # Prepare current class in batch
                    batch[s] = self.indexes[class_label][sample_idxs]        

                # Construct batch - shuffle the generated batch - add query as last element
                batch = batch[torch.randperm(batch_size-1)]
                batch = torch.cat((
                    batch, 
                    torch.tensor([query_idx], dtype=torch.long)
                ))
                yield batch


    def decode_batch(self, batch_labels, batch_classes):

        """ 
        Returns the indexes of support and query sets for each class
        """

        # Last element is query, rest are support
        support_idxs = [
            batch_labels[:-1].eq(c).nonzero().squeeze(1)
            for c in batch_classes
        ]

        query_idxs = [ torch.tensor([len(batch_labels)-1]) ]
        return support_idxs, query_idxs

    
    def __len__(self):
        
        """
        Returns the number of iterations (episodes) per epoch
        """
        return self.iterations
