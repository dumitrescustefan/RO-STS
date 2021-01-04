![License](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)

### RO-STS: Romanian Semantic Text Similarity Dataset 

This folder contains the two types of files released:

#### 1. Text similarity

In the ``text-similarity`` folder there are 4 tsv files:

* ``ro-sts.tsv`` containing all the sentence pairs in the dataset (this file contains train+dev+test released as a single file)
* ``ro-sts.train.tsv`` containing the train portion of the dataset
* ``ro-sts.dev.tsv`` containing the development portion of the dataset
* ``ro-sts.test.tsv`` containing the test portion of the dataset

The train/dev/test splits are identical to the original English STS corpus splits.

All files have the same tab-separated format:
``` 
Sentence 1 [TAB] Sentence 2 [TAB] Similarity score
```
where ``Similarity score`` is a floating point number from 0 to 5, with 5 being the maximum similarity.

#### 2. Parallel corpus

To be constructed.