![Python](https://img.shields.io/badge/Python-3-brightgreen) 

## RO-STS: Romanian Semantic Textual Similarity Dataset - Baseline Models

We provide the following models to provide an initial baseline score on this Text Similarity task. All results obtained here are reproducible by running each evaluation as detailed below.

Here are the results after running each model 10 times and averaging scores. 

| Model                       	| # of parameters  	| Dev-set Pearson 	| Dev-set Spearman 	| Test-set Pearson 	| Test-set Spearman 	|
|-----------------------------	|------------------	|-----------------	|------------------	|------------------	|-------------------	|
| RNN                         	|       16.7M      	|      0.7342     	|      0.7349      	|      0.6744      	|       0.6662      	|
| Romanian BERT v1 (uncased)  	|       124M       	|      0.8453     	|      0.8417      	|    **0.8156**    	|     **0.8075**    	|
| Romanian BERT v1 (cased)    	|       124M       	|      0.8477     	|      0.8447      	|      0.7985      	|       0.7897      	|
| Multilingual BERT (uncased) 	|       167M       	|      0.8237     	|      0.8235      	|      0.7690      	|       0.7650      	|
| Multilingual BERT (cased)   	|       167M       	|      0.8071     	|      0.8077      	|      0.7664      	|       0.7641      	|
| Distill BERT (cased)          |        81M        |      0.7737       |      0.7726       |      0.7253       |       0.7167        |
| Readerbench/RoGPT-base        |       124M        |      0.8210       |      0.8185       |      0.7848       |       0.7729        |
| Readerbench/RoGPT-medium      |       354M        |      0.8400       |      0.8387       |      0.7954       |       0.7867        |
| xlm-roberta-base              |       278M        |      0.8215       |      0.8233       |      0.7783       |       0.7756        |


If you would like to run these models yourself please do a ``pip install -r requirements.txt`` from this folder.

#### 1. Transformer baseline

The transformer model encodes each sentence separately, then does a mean-pooling of the output token vectors. The two resulting sentence representations are compared using the cosine sim score. The 0-5 interval was reduced to 0-1 (to be compatible with the cosine), and the MSE loss was used. This is the standard recipe for STS/NLI tasks, and we provide it here as a baseline.

The transformer baseline was run with the Romanian BERT (``--model_name dumitrescustefan/bert-base-romanian-uncased-v1``) and with the multilingual BERT (``--model_name bert-base-multilingual-uncased``). Both the cased and uncased versions of each transformer were used for comparison.    

Run the model as ``python transformer_model.py`` with the appropriate ``--model_name`` parameter to obtain the results in the table above. Use ``--help`` to see available parameters that you can change. For example, to run a single iteration, use ``--experiment_iterations 1``.

#### 2. Recurrent Neural Network baseline

The RNN model uses bidirectional stacked LSTMs to encode each sentence. Each sentence representation is passed through an attention layer. Similarly to the transformer model, the cosine similarity function is used, as well as the same MSE loss. Please note that while the transformer model comes with its own tokenizer/vocab, here we need to create it first, and this is exactly what this model will do if it is not already created. It creates a WordPiece tokenizer based on the full dataset. 

Run the model as ``python rnn_model.py`` without any parameters to obtain the results in the table above. Use ``--help`` to see available parameters that you can change. For example, to run this model on a CPU use ``--gpus 0``, as by default, the model will attempt to use one GPU.
