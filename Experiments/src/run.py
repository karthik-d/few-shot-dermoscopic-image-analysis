from data.config import config as data_config
from prototypical.config import config as proto_config 

from prototypical import trainer, tester_exhaustive, tester_exhaustive_extended
from classifier_nw import trainer as cnw_trainer
from classifier_nw import tester as cnw_tester 

from data import split_train_test

# trainer.train()
# tester_exhaustive.test()
# tester_exhaustive_extended.test()

cnw_trainer.train()
cnw_tester.test()

# split_train_test.split_data_all_classes()
# split_train_test.split_test_classes()

# run()
