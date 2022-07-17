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

        classes = torch.unique(target_cpu)
        n_classes = len(classes)

        (support_idxs, query_idxs) = sampler.decode_batch(
            batch_labels=target_cpu, 
            batch_classes=classes
        )

        support_idxs = torch.stack(support_idxs).view(-1).long()
        support_samples = input_cpu[support_idxs]
        support_truths = target_cpu[support_idxs]
        n_support = len(support_idxs[0])

        query_idxs = torch.stack(query_idxs).view(-1).long()
        query_samples = input_cpu[query_idxs]
        query_truths = target_cpu[query_idxs]
        n_query = len(query_idxs[0])

        query_preds = classifier(
            support_samples,
            support_truths,
            query_samples
        )

        acc_val = query_truths.eq(query_preds).float().mean()    

        if not get_prediction_results:
            return acc_val
        else:
            return acc_val, (query_preds, query_truths)

    # Return classifier with sampler and type bound
    return local_classifier