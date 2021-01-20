![License](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)

## RO-STS: Romanian Semantic Textual Similarity Dataset 

This folder contains the two types of files released:

### 1. Textual similarity

In the ``text-similarity`` folder there are 3 *tsv* files:

* ``RO-STS.train.tsv`` containing the train portion of the dataset
* ``RO-STS.dev.tsv`` containing the development portion of the dataset
* ``RO-STS.test.tsv`` containing the test portion of the dataset

The train/dev/test splits are identical to the original English STS corpus splits.

All files have the same tab-separated format:
``` 
Similarity score [TAB] Sentence 1 [TAB] Sentence 2 
```
where ``Similarity score`` is a floating point number from 0 to 5, with 5 being the maximum similarity.

For example:

```
1.5	Un bărbat cântă la harpă.	Un bărbat cântă la claviatură.
1.8	O femeie taie cepe.	O femeie taie tofu.
3.5	Un bărbat merge pe o bicicletă electrică.	Un bărbat merge pe bicicletă.
2.2	Un bărbat cântă la tobe.	Un bărbat cântă la chitară.
2.2	Un bărbat cântă la chitară.	O doamnă cântă la chitară.
```

### 2. Parallel corpus (RO-EN)

The parallel corpus is found in the ``ro-en`` folder, and contains 6 files:

* ``RO-STS.train.ro`` and ``RO-STS.train.en`` representing the train portion of the dataset
* ``RO-STS.dev.ro`` and ``RO-STS.dev.en`` representing the development portion of the dataset
* ``RO-STS.test.ro`` and ``RO-STS.test.en`` representing the test portion of the dataset

The files are raw text, one-sentence per line, each file pair containing corresponding ``ro`` - ``en`` sentences.