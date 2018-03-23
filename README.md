# kaggle_toxic_comment
This repositroy is the Code for Kaggle Competition: [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### Dataset
[The dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) is available via Kaggle competition
Unzip all data and put them into './input/' folder

### Requirements
Anaconda == 5.1.0
python == 3.6
tensorflow == 1.6.0
keras == 2.1.5


### Pretrained Word Embeddings: 
  * [fastText: wiki.en.bin](https://fasttext.cc/docs/en/english-vectors.html)
  * [GloVe: glove.840B.300d](https://nlp.stanford.edu/projects/glove/) 

### Models (best private score shown): 
   - rnn    (*0.9860 private*,	0.9859 public)
   - rnn-cnn (*0.9858 private*,	0.9863 public)
   - rnn-rnn (*0.9856 private*,	0.9861 public)
   - rnn-rnn-cnn (*0.9826 private*,	0.9835 public)
	
