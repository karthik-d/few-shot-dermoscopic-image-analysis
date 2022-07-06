from data.config import config as data_config
from prototypical.config import config as proto_config 

from prototypical import trainer, tester

from data import split_train_test

# trainer.train()
# tester.test()

split_train_test.split_data_all_classes()
split_train_test.split_test_classes()

# run()
