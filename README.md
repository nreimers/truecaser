# Language Independet Truecaser for Python
This is an implementation of a trainable Truecaser for Python.

A truecaser converts a sentence where the casing was lost to the most probable casing. Use cases are sentences that are in all-upper case, in all-lower case or in title case.


# Models
This repository contains three models.

* **GreedyTruecaser**: Trains dictionary with the casing for unigrams, bigrams and trigrams. Words are then replaced with the most likely casing. This model is the fastest to train and at inference and the model size is fairly small.
* **StatisticalTruecaser**: This class trains a statistical model, that computes the probability for all possible casing combinations. Statistics are based on uni-, bi- and trigram frequencies. Training and inference is fast, however, the models can become quite large as no pruning is performed.
* **NeuralTruecaser**: Trains an LSTM-network for truecasing. This network gives good performance on small training sets. However, train and inference time is extremely slow.

My personal recommendation would be the **GreedyTruecaser**. It is fast to train and at inference and the performance is comparable to the more complex models.


# Requirements
The Code was written for Python 2.7 and requires NLTK 3.0.

The NeuralTruecaser requires Keras 1.x and Theano / Tensorflow as backend.

# Train / Development / Test Files
The `Evaluate.py` uses train, development and test files for training / evaluating the truecase models. You can find training / testing files for English / German / Turkish here:
[public.ukp.informatik.tu-darmstadt.de/reimers/2017_Wikipedia_Truecase/wikipedia_train_dev_test_files.zip](Download wikipedia files for truecasing).

Unzip those to the `wikipedia/` folder.

# Run the Code
 

# Train your own Truecaser



