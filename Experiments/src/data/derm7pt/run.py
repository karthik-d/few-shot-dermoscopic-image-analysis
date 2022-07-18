from dataset import Derm7PtDataset, Derm7PtDatasetGroupInfrequent
import os
from matplotlib import pyplot as plot
import pandas as pd

dir_release = '/home/miruna/Skin-FSL/Derm7pt/release_v0'
dir_meta = os.path.join(dir_release, 'meta')
dir_images = os.path.join(dir_release, 'images')

meta_df = pd.read_csv(os.path.join(dir_meta, 'meta.csv'))
train_indexes = list(pd.read_csv(os.path.join(dir_meta, 'train_indexes.csv'))['indexes'])
valid_indexes = list(pd.read_csv(os.path.join(dir_meta, 'valid_indexes.csv'))['indexes'])
test_indexes = list(pd.read_csv(os.path.join(dir_meta, 'test_indexes.csv'))['indexes'])

# The full dataset before any grouping of the labels.
derm_data = Derm7PtDataset(dir_images=dir_images, 
                        metadata_df=meta_df.copy(), # Copy as is modified.
                        train_indexes=train_indexes, valid_indexes=valid_indexes, 
                        test_indexes=test_indexes)

derm_data_group = Derm7PtDatasetGroupInfrequent(dir_images=dir_images, 
                                             metadata_df=meta_df.copy(), # Copy as is modified.
                                             train_indexes=train_indexes, 
                                             valid_indexes=valid_indexes, 
                                             test_indexes=test_indexes)

train_derm_paths = derm_data_group.get_img_paths(data_type='train', img_type='derm')
train_clinic_paths = derm_data_group.get_img_paths(data_type='train', img_type='clinic')
train_labels = derm_data_group.get_labels(data_type='train', one_hot=False)

# Get the dermatology and clinic test images and corresponding labels.
test_derm_paths = derm_data_group.get_img_paths(data_type='test', img_type='derm')
test_clinic_paths = derm_data_group.get_img_paths(data_type='test', img_type='clinic')
test_labels = derm_data_group.get_labels(data_type='test', one_hot=False)

val_derm_paths = derm_data_group.get_img_paths(data_type='valid', img_type='derm')
val_clinic_paths = derm_data_group.get_img_paths(data_type='valid', img_type='clinic')
val_labels = derm_data_group.get_labels(data_type='valid', one_hot=False)

# print(derm_data.df.diagnosis.unique())
print(train_labels['DIAG'].value_counts())
print(val_labels['DIAG'].value_counts())
print(test_labels['DIAG'].values_counts())
label_names = derm_data_group.get_label_by_abbrev('DIAG').abbrevs.values
print(label_names)
print(derm_data.get_label_abbrevs('DIAG'))
print(derm_data.get_label_nums('DIAG'))
print(derm_data.get_column_name_numeric('DIAG'))