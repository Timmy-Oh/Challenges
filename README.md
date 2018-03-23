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


| Model	| Embeddings | Public | Private | Public | Local |
|:------ |:---------- | ------- | ------ | ----- |
| RNN	| fasttext	| 0.9843	| 0.9850	| 0.9896|

   - |rnn    |(*0.9845 private*,	|0.9859 public)|
   - rnn-cnn (*0.9858 private*,	0.9863 public)
   - rnn-rnn (*0.9856 private*,	0.9861 public)
   - rnn-rnn-cnn (*0.9826 private*,	0.9835 public)
	


| Model	| Embeddings | Private | Public | Local |
|:------ |:---------- | ------- | ------ | ----- |
|  |
| CapsuleNet	| fasttext	| 0.9855	| 0.9867	| 0.9896|
| CapsuleNet	| glove	| 0.9860 	| 0.9859	| 0.9899|
| CapsuleNet	| lexvec	| 0.9855	| 0.9858	| 0.9898|
| CapsuleNet	| toxic	| 0.9859	| 0.9863	| 0.9901|
|  |
| RNN Version 2	| fasttext	| 0.9856	| 0.9864	| 0.9904|
| RNN Version 2	| glove	| 0.9858 	| 0.9863	| 0.9902|
| RNN Version 2	| lexvec	| 0.9857	| 0.9859	| 0.9902|
| RNN Version 2	| toxic	| 0.9851	| 0.9855	| 0.9906|
|  |
| RNN Version 1	| fasttext	| 0.9853	| 0.9859	| 0.9898|
| RNN Version 1	| glove	| 0.9855	| 0.9861	| 0.9901|
| RNN Version 1	| lexvec	| 0.9854	| 0.9857	| 0.9897|
| RNN Version 1	| toxic	| 0.9856 | 0.9861	| 0.9903|
|  |
| 2 Layer CNN	| fasttext	| 0.9826	| 0.9835	| 0.9886|
| 2 Layer CNN	| glove 	| 0.9827	| 0.9828	| 0.9883|
| 2 Layer CNN	| lexvec	| 0.9824	| 0.9831	| 0.9880|
| 2 Layer CNN	| toxic	| 0.9806	| 0.9789	| 0.9880|
|  |
| SVM with NB features	| NA	| 0.9813	| 0.9813	| 0.9863|
