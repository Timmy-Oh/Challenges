import numpy as np
import codecs

# func for load embedding weights
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def load_emb_model(embedding_path):
    return dict(get_coefs(*o.strip().split(" ")) for o in codecs.open(embedding_path, "r", "utf-8" ))