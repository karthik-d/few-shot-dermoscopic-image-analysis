## Dataset Description

### Malignant and Benign
- Malignant Classes: `MEL`, `BCC`, `AKIEC` 
- Benign Classes: `NV`, `BKL`, `DF`
- Benign / Malignant: `VASC`

### Distribution

![Data Distribution](/assets/ISIC18_T3_Distribution.png)


## Experiment Methods

- `L-way n-shot`: `L` classes are randomly sampled from `support domain` into the support set. For each class, `n` images are chosen and placed into the support set. A different parameter, `m` determines the number of images per class that will be chosen for the query set. 

- `Prototypical Networks`: During training, episodes consisting of a support set S and a query set Q are sampled as described earlier. Then, a prototype for each class `k` is computed as the mean embedding of the samples
from the support set belonging to that class.

- [Reference: Episodic Learning](https://openreview.net/pdf?id=bJaZ8leI0QJ)

## Dataset Phases

### DS Phase 1

- All classes are part of both, train and test sets
- Split randomly using `src/split_train_test` in 70-30 ratio (ratio is per-class)
    - Use the function `split_data_all_classes()`

#### Experiment 1

##### Configuration
```
TRAINING CONFIGURATION
Class Names:              ([ MEL, NV, BCC, AKIEC, BKL, DF, VASC ])
Class Distribution: tensor([ 835, 5029,  386,  246,  825,   87,  107])

TESTING CONFIGURATION
Class Names:        ([ MEL, NV, BCC, AKIEC, BKL, DF, VASC ])
Class Distribution: ([ 278, 1676,  128,   81,  274,   28,   35])
Support Domain: tensor([0, 1, 2, 3, 4, 5, 6])
Forced Support: tensor([])
Query Domain:   tensor([0, 1, 2, 3, 4, 5, 6])
```

##### Results
![Result: Confusion Matrix](/assets/confusion-matrix/ds-phase-1_exp-1.jpg)

### DS Phase 2

- Most mis-classified classes are removed from the train set, and moved to the test set
- Specifically, `MEL`, `NV` and `BCC` are made the test set, and the model is trained only using the other 4 classes.
- Split deterministically using `src/split_train_test` in 70-30 ratio (ratio is per-class)
    - Use the function `split_test_classes()`
- This tests the strength of the similarity function learnt by the model

#### Experiment 1

##### Configuration
```
TRAINING CONFIGURATION
Class Names:              ([ AKIEC, BKL, DF, VASC ])
Class Distribution: tensor([ 327, 1099,  115,  142])

TESTING CONFIGURATION
Class Names:        ([ MEL, NV, BCC ])
Class Distribution: ([1113, 6705,  514])
Support Domain: tensor([0, 1, 2])
Forced Support: tensor([])
Query Domain:   tensor([0, 1, 2])
```

##### Results
![Result: Confusion Matrix](/assets/confusion-matrix/ds-phase-2_exp-1.jpg)

### DS Phase 3

- Complete data is split in 70-30 ratio (per-class ratio) into train and test sets
- Most mis-classified classes are removed from the train set, and moved to isolation (these are never seen by the trainer)
- Specifically, `MEL`, `NV` and `BCC` are moved to isolation set from test set, and the model is trained only using the other 4 classes
- This tests the strength of the similarity function learnt by the model
- Split randomly using `src/split_train_test` in 70-30 ratio (ratio is per-class)
    - Use the functions `split_data_all_classes()` and `split_test_classes()`, in succession


#### Experiment 1

##### Configuration
```
TRAINING CONFIGURATION
Class Names:              ([ AKIEC, BKL, DF, VASC ])
Class Distribution: tensor([229, 770,  81, 100])

TESTING CONFIGURATION
Class Names:        ([ MEL, NV, BCC, AKIEC, BKL, DF, VASC ])
Class Distribution: ([ 333, 2011,  154,   98,  329,   34,   42])
Support Domain: tensor([0, 1, 2, 3, 4, 5, 6])
Forced Support: tensor([])
Query Domain:   tensor([0, 1, 2])
```

- 2-way, 3-shot testing
- The test set contains all 7 classes
- When sampling, query set is populated only with one of `MEL`, `NV` and `BCC`. Support set, however, can contain any one of the seven classes

##### Results
![Result: Confusion Matrix](/assets/confusion-matrix/ds-phase-3_exp-1.jpg)

#### Experiment 2

##### Configuration
```
TRAINING CONFIGURATION
Class Names:              ([ AKIEC, BKL, DF, VASC ])
Class Distribution: tensor([229, 770,  81, 100])

TESTING CONFIGURATION
Class Names:        ([ MEL, NV, BCC, AKIEC, BKL, DF, VASC ])
Class Distribution: ([ 333, 2011,  154,   98,  329,   34,   42])
Support Domain: tensor([0, 1, 2, 3, 4, 5, 6])
Forced Support: tensor([])
Query Domain:   tensor([0, 1, 2])
```

- 3-way, 3-shot testing
- The test set contains all 7 classes
- When sampling, query set is populated only with one of `MEL`, `NV` and `BCC`. Support set, however, can contain any one of the seven classes

##### Results
![Result: Confusion Matrix](/assets/confusion-matrix/ds-phase-3_exp-2.jpg)



#### Experiment 3

##### Configuration
```
TRAINING CONFIGURATION
Class Names:              ([ AKIEC, BKL, DF, VASC ])
Class Distribution: tensor([229, 770,  81, 100])

TESTING CONFIGURATION
Class Names:        ([ MEL, NV, BCC, AKIEC, BKL, DF, VASC ])
Class Distribution: ([ 333, 2011,  154,   98,  329,   34,   42])
Support Domain: tensor([0, 1, 2, 3, 4, 5, 6])
Forced Support: tensor([ 1 ])
Query Domain:   tensor([0, 1, 2])
```

- 3-way, 3-shot testing
- The test set contains all 7 classes
- When sampling, query set is populated only with one of `MEL`, `NV` and `BCC`. Support set, however, can contain any one of the seven classes
- As an additional constraint, to study the impact of misclassification induced by `NV`, it is always included in the support set

##### Results
![Result: Confusion Matrix](/assets/confusion-matrix/ds-phase-3_exp-3.jpg)


### DS Phase 4

- Complete data is split in 70-30 ratio (per-class ratio) into train and test sets
- Classes with the least number of images-per-class are removed from the train set, and moved to isolation (these are never seen by the trainer)
- Specifically, `AKIEC`, `VASC` and `DF` are moved to isolation set from test set, and the model is trained only using the other 4 classes
- This tests the ability of the model to learn the "similarity" function from a large dataset, and successfully use it to distinguish between classes that have limited annotated data available
- Split randomly using `src/split_train_test` in 70-30 ratio (ratio is per-class)
    - Use the functions `split_data_all_classes()` and `split_test_classes()`, in succession

#### Experiment 1

##### Configuration
```
TRAINING CONFIGURATION
Class Names:              ([ MEL, NV, BCC, BKL ])
Class Distribution: tensor([ 780, 4694,  360,  770])


TESTING CONFIGURATION
Class Names:        ([ MEL, NV, BCC, AKIEC, BKL, DF, VASC ])
Class Distribution: ([ 333, 2011,  154,   98,  329,   34,   42])
Support Domain: tensor([0, 1, 2, 3, 4, 5, 6])
Forced Support: tensor([ 1 ])
Query Domain:   tensor([3, 5, 6])
```

- 3-way, 3-shot testing
- The test set contains all 7 classes
- When sampling, query set is populated only with one of `MEL`, `NV` and `BCC`. Support set, however, can contain any one of the seven classes
- As an additional constraint, to study the impact of misclassification induced by `NV`, it is always included in the support set

##### Results

![Result: Confusion Matrix](/assets/confusion-matrix/ds-phase-4_exp-1.jpg)