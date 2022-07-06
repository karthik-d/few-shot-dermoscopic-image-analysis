from data.config import config as data_config
from prototypical.config import config as proto_config 

from prototypical import trainer, tester_exhaustive, tester_exhaustive_extended

from data import split_train_test

# trainer.train()
tester_exhaustive.test()
# tester_exhaustive_extended.test()

# split_train_test.split_data_all_classes()
# split_train_test.split_test_classes()

# run()
