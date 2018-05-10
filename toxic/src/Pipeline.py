import pandas as pd
import numpy as np
import codecs
import re
import string
import os

#===============keras ==============
from keras.preprocessing import text, sequence

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def load_emb_model(embedding_path):
    return dict(get_coefs(*o.strip().split(" ")) for o in codecs.open(embedding_path, "r", "utf-8" ))

def load_data_2path(emb_model,
             filepath_train = "./input/train.csv", 
             filepath_test = "./input/test.csv", 
             embed_size = 300,
             max_features = 100000,
             maxlen = 180
            ):

    DOC_Column = "comment_text"
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    ###load data    
    train = pd.read_csv(filepath_train)
    test = pd.read_csv(filepath_test)
    print("=== Data is loaded")

    list_sentences_train = train[DOC_Column].fillna('UNK').values
    list_sentences_test = test[DOC_Column].fillna('UNK').values
    y = train[list_classes].values

    preprocessed_train = list_sentences_train.tolist()
    preprocessed_test = list_sentences_test.tolist()
    
    tokenizer = text.Tokenizer(num_words =max_features)
    tokenizer.fit_on_texts(preprocessed_train + preprocessed_test)

    list_tokenized_train = tokenizer.texts_to_sequences(preprocessed_train)
    list_tokenized_test = tokenizer.texts_to_sequences(preprocessed_test)

    X_t_pre = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, truncating='pre')
    X_t_post = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, truncating='post')
    
    X_te_pre = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, truncating='pre')
    X_te_post = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, truncating='post')
    print("=== Data is preprocessed")
    
    X_t = [X_t_pre, X_t_post]
    X_te = [X_te_pre, X_te_post]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
#     embedding_matrix = np.zeros((nb_words, embed_size))
    embedding_matrix = np.random.normal(0.001, 0.4, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_features: continue
        try:
            embedding_vector = emb_model.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        except: 
            pass
    print("=== Embedding Matrix is loaded")

    return X_t, y, X_te, embedding_matrix