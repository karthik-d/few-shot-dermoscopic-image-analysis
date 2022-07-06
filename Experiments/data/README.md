## Dataset Phases

### DS Phase 1

- All classes are part of both, train and test sets
- Split randomly using `src/split_train_test` in 75-25 ratio (ratio is per-class)


### DS Phase 2

- Most mis-classified classes are removed from the train set, and moved to the test set
- Specifically, `MEL`, `NV` and `BCC` are made the test set, and the model is trained only using the other 4 classes
- This tests the strength of the similarity function learnt by the model


### DS Phase 3

- Complete data is split in 70-30 ratio (per-class ratio) into train and test sets
- Most mis-classified classes are removed from the train set, and moved to isolation (these are never seen by the trainer)
- Specifically, `MEL`, `NV` and `BCC` are moved to isolation set from test set, and the model is trained only using the other 4 classes
- This tests the strength of the similarity function learnt by the model


#### Experiment 1

- 2-way, 3-shot
- The test set contains all 7 classes
- When sampling, query set is populated only with one of `MEL`, `NV` and `BCC`. Support set, however, can contain any one of the seven classes

#### Experiment 2

```
TRAINING CONFIGURATION
Class Names:              ([ AKIEL, BKL, DF, VASC ])
Class Distribution: tensor([229, 770,  81, 100])

TESTING CONFIGURATION
Class Names:       ([ MEL, NV, BCC, AKIEL, BKL, DF, VASC ])
Class Disribution: ([ 333, 2011,  154,   98,  329,   34,   42])
Support Domain: tensor([0, 1, 2, 3, 4, 5, 6])
Forced Support: tensor([ 1 ])
Query Domain: tensor([0, 1, 2])
```

- 3-way, 3-shot
- The test set contains all 7 classes
- When sampling, query set is populated only with one of `MEL`, `NV` and `BCC`. Support set, however, can contain any one of the seven classes
- As an additional constraint, to study the impact of misclassification induced by `NV`, it is always included in the support set

