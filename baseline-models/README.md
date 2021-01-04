![Python](https://img.shields.io/badge/Python-3-brightgreen) 

### RO-STS: Romanian Semantic Text Similarity Dataset Baseline Models

We provide the following models to provide an initial baseline score on this Text Similarity task. All results obtained here are reproducible by running each evaluation as detailed below.

Here are the results after running each model 10 times and averaging scores. 

|                   	| Transformer 	| RNN 	|
|-------------------	|:-----------:	|:---:	|
| Dev-set Pearson   	|      0      	|  0  	|
| Dev-set Spearman  	|      0      	|  0  	|
| Test-set Pearson  	|      0      	|  0  	|
| Test-set Spearman 	|      0      	|  0  	|

If you would like to run these models yourself please do a ``pip install -r requirements.txt`` from this folder. Do this in a clean virtual env as the versions of pytorch, transformers, etc., are frozen to provide as much future-proofing for these scripts as possible.
#### 1. Transformer baseline

The transformer model uses the Romanian BERT model to encode each sentence separately, then does a mean-pooling of the output token vectors. The two resulting sentence representations are compared using the cosine sim score. The 0-5 interval was reduced to 0-1 (to be compatible with the cosine), and the MSE loss was used. This is the standard recipe for STS/NLI tasks, and we provide it here as a baseline.

Run the model as ``python transformer_model.py`` without any parameters to obtain the results in the table above. Use ``--help`` to see available parameters that you can change. For example, to run a single iteration, use ``--experiment_iterations 1``.

#### 2. Recurrent Neural Network baseline

The RNN model uses bidirectional stacked LSTMs to encode each sentence. Each sentence representation is passed through an attention layer. Similarly to the transformer model, the cosine similarity function is used, as well as the same MSE loss. Please note that while the transformer model comes with its own tokenizer/vocab, here we need to create it first, and this is exactly what this model will do if it is not already created. It creates a WordPiece tokenizer based on the full dataset. 

Run the model as ``python rnn_model.py`` without any parameters to obtain the results in the table above. Use ``--help`` to see available parameters that you can change. For example, to run this model on a CPU use ``--gpus 0``, as by default, the model will attempt to use one GPU.
