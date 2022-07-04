## Dataset Phases

### DS Phase 1

- All classes are part of both, train and test sets
- Split randomly using `src/split_train_test` in 75-25 ratio (ratio is per-class)


### DS Phase 2

- Most mis-classified classes are removed from the train set, and moved to the test set
- Specifically, `MEL`, `NV` and `BCC` are made the test set, and the model is trained only using the other 4 classes
- This tests the strength of the similarity function learnt by the model

