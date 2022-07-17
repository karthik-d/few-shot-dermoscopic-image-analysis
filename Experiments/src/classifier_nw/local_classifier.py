import torch
import numpy as np

from classifiers import logistic_classifier


def get_local_classifier(classifier_name='LR', sampler=None):

    if classifier_name == 'LR':
        classifier = logistic_classifier

    def local_classifier(input, target, get_prediction_results=False):

        """
        Args:
        - input: the model output for a batch of samples
        - target: ground truth for the above batch of samples
        - get_prediction_results: If True, a third return value corresponds to 
                                    (predicted_labels, actual_labels) 
        """

        target_cpu = target.to('cpu')
        input_cpu = input.to('cpu')

        classes = np.unique(target_cpu.detach().numpy())
        n_classes = len(classes)

        (support_idxs, query_idxs) = sampler.decode_batch(
            batch_labels=target_cpu, 
            batch_classes=classes
        )

        n_support = len(support_idxs[0])
        support_idxs = torch.stack(support_idxs).view(-1).long()
        support_samples = input_cpu[support_idxs]
        support_truths = target_cpu[support_idxs]

        n_query = len(query_idxs[0])
        query_idxs = torch.stack(query_idxs).view(-1).long()
        query_samples = input_cpu[query_idxs]
        query_truths = target_cpu[query_idxs]

        query_preds = classifier.fit_predict(
            support_samples.detach().numpy(),
            support_truths.detach().numpy(),
            query_samples.detach().numpy()
        )

        acc_val = np.count_nonzero(query_truths.detach().numpy()==query_preds)/len(query_truths)  
        if not get_prediction_results:
            return acc_val
        else:
            return (
                acc_val, 
                (
                    torch.LongTensor(query_preds),
                    query_truths
                )
            )

    # Return classifier with sampler and type bound
    return local_classifier