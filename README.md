![License](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)

(UNDER CONSTRUCTION) 

# RO-STS: Romanian Semantic Text Similarity Dataset 

This dataset is the Romanian version of the [STS](https://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) dataset.
It is a high-quality translation of the aforementioned dataset, containing 7804 pairs of sentences with their similarity score. The dataset respects the same split: 5215 train, 1316 dev and 1273 test sentence pairs.

### Dataset format

The dataset is offered in two flavours:

##### 1. Text similarity dataset

TBD.

##### 2. Parallel corpus 

TBD.

### Baseline evaluation

We provide 2 baselines for this dataset, a transformer based model and a recurrent neural network model. Both models were trained on the train set until the Pearson score did not improve on the dev set, and results are reported on the test set.

Table with results coming soon.

For more details on how to reproduce these scores please check out the [detailed evaluation page](baseline/README.md).

### Creation process

The dataset was created in three steps:

1. Automatic translation with Google's translation service.
2. Correction round by a person that rectified all errors resulting from the automatic translation - and there were plenty.
3. Validation by a different person that double-checked the translation.

This process has ensured the high-quality translation of this dataset.

Here are the annotators/contributors, alphabetically listed:
* Adriana STAN
* Andrei PRUTEANU
* Andrei-Marius AVRAM
* Beáta LŐRINCZ
* Madalina Chitez 
* Mihai ILIE
* Petru REBEJA
* Razvan PASCANU
* Stefan Daniel DUMITRESCU
* Viorica PATRAUCEAN

### Licensing

This work, like it's [original](https://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark), is licensed as [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/). That means you're free to do anything you want with it, as long as you keep the same license.

### Citation

Coming soon.

