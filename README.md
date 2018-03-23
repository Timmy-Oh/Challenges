# kaggle_toxic_comment
the Code for Kaggle Competition: [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

**475th place in Competition**

## Dataset:
[The dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) is available via Kaggle competition
Unzip all data and put them into './input/' folder

## Requirements:
 * Anaconda == 5.1.0
 * python == 3.6
 * tensorflow == 1.6.0
 * keras == 2.1.5


## Pretrained Word Embeddings: 
  * [FastText: crawl-300d-2M](https://github.com/facebookresearch/fastText/blob/master/docs/english-vectors.md) \[[download](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip)\]
  * [GloVe: glove.840B.300d](https://nlp.stanford.edu/projects/glove/) \[[download](http://nlp.stanford.edu/data/glove.840B.300d.zip)\]

## Models: 


| Model	| Embeddings | Public | Private	|
|:------ |:---------- | ------- | ------ |
| |
| RNN	| fasttext	| 0.9850	| 0.9843	|
| RNN-CNN | fasttext	| 0.9846	| 0.9842	|

