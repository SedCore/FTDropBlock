import sys
import os
import logging
from braindecode.datasets import MOABBDataset

### Suppress undesirable print statements temporarily
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


### Load the dataset
def load_dataset(dataset_name):
    if dataset_name=="BCICIV_2a":
        return MOABBDataset(dataset_name="BNCI2014_001", subject_ids=None)
    else:
        raise ValueError(f'Dataset {dataset_name} is unknown.')
