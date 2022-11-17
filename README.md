# NLP Adapter challenge: Aligning translations by embedding similarity

The idea of this notebook (run in google colab) is to use pre-trained models to find translation Czech-English pairs.

## Idea

We want to find a method that can find corresponding translation pairs from similarities of text embeddings.
In particular, we want to align CS<>EN translations of the [TED dataset](https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/).

For this task, [Sentence Transformers](https://arxiv.org/abs/1908.10084) would be more useful, since they find embeddings for sentences, and not word-by-word.
However, I restrict to using simpler architectures.

## What did I try to do?

- From a [TED dataset](https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/), compute sentence embeddings using [BERT](https://arxiv.org/abs/1810.04805). 
- Find a similarity metric for embeddings (for matching translations)
- Train [Adapters](https://arxiv.org/pdf/1902.00751.pdf) on the [BERT models](https://arxiv.org/abs/1810.04805) to fine-tune to this task

## Models used

I use pretrained [Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) models, to compute sentence embeddings for Czech and English sentences.

![BERT](figs/bert.JPG)

We need models that contain information of the Czech language.
Therefore models such as [bert-base-cased]((https://huggingface.co/bert-base-cased?text=Paris+is+the+%5BMASK%5D+of+France)) do not work.
I use [bert-base-multilingual-cased](https://huggingface.co/bert-base-cased?text=The+goal+of+life+is+%5BMASK%5D)  trained in a self-supervised fashion for the tasks of Masked Language Modeling and Next Sentence Prediction.

We expect this models to contain valuable information on the meaning of different sentences.

I use the intermediate representation on the pre-trained network of the given sentences as embeddings.
In particular, I use the last layer, which should contain higher-level information on the meaning of the sentences.


The embeddings of a BERT (naive) transformer contain contextualized work embeddings; not directly sentence embeddings. 
I use mean-pooling to convert this embedding into “sentence-embeddings”.

## Similarity metrics

We may use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) or [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error).


The cosine similarity measures how much to embeddings point to the same direction.
The mean squared error is the euclidean distance of the embeddings.

We expect that sentences with the same meaning (being translations one of each other) point to the same dimension and their embeddings are close.

This, however, depends on how the pre-trained model was trained, for which task, on the data using for training, ….

For 100 samples of the data, we get that the cosine similarity on translation pairs is 0.491, and the MSE is 0.218
If we randomly change the ordering, the cosine similarity becomes 0.176 and the MSE distance 0.264.

## Towards a better metric

A performance metric can consist in checking if, for a row, the maximum (or minimum) value is in the diagonal. If it is, it is counted as 1, and else as 0. 
The mean of this value can be seen as a measure of accuracy.

[Metrics Comparison](figs/metrics_comp.JPG)

For 100 samples of the TED dataset, the performance (“accuracy”) metric is 0.68 for CosSim and 0.55 for MSE.
A random selection would lead an "accuracy" metric of 0.01.

## PCA representations

A [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) representation of the embeddings can give interesting visual representation of the data.

[PCA example](figs/pca_example.JPG)




