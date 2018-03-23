#===============keras ==============
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, add
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.layers import K, Activation
from keras.engine import Layer
    
def get_model_rnn(
                  embedding_matrix, cell_size = 80, cell_type_GRU = True,
                  maxlen = 180, max_features = 100000, embed_size = 300,
                  prob_dropout = 0.2, emb_train = False
                 ):
    
    inp_pre = Input(shape=(maxlen, ), name='input_pre')
    inp_post = Input(shape=(maxlen, ), name='input_post')

    ##pre
    x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_pre)
    x1 = SpatialDropout1D(prob_dropout)(x1)
    
    if cell_type_GRU:
        x1 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1)
    else :
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
    
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    
    ##post
    x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
    x2 = SpatialDropout1D(prob_dropout)(x2)
    
    if cell_type_GRU:
        x2 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2)
    else :
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
    
    avg_pool2 = GlobalAveragePooling1D()(x2)
    max_pool2 = GlobalMaxPooling1D()(x2)
    
    ##merge
    conc = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=[inp_pre, inp_post], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_crossentropy', 'accuracy'])

    return model

def get_model_rnn_cnn(
                      embedding_matrix, cell_size = 80, cell_type_GRU = True,
                      maxlen = 180, max_features = 100000, embed_size = 300,
                      prob_dropout = 0.2, emb_train = False,
                      filter_size=128, kernel_size = 2, stride = 1
                      ):
    inp_pre = Input(shape=(maxlen, ), name='input_pre')
    inp_post = Input(shape=(maxlen, ), name='input_post')
    
    ##pre
    x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_pre)
    x1 = SpatialDropout1D(prob_dropout)(x1)
    
    if cell_type_GRU:
        x1 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1)
    else :
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
    
    x1 = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1)
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    
    ##post
    x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
    x2 = SpatialDropout1D(prob_dropout)(x2)
    
    if cell_type_GRU:
        x2 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2)
    else :
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
    
    x2 = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    max_pool2 = GlobalMaxPooling1D()(x2)
    
    ##merge
    conc = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    outp = Dense(6, activation="sigmoid")(conc)
    
    
    model = Model(inputs=[inp_pre, inp_post], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_crossentropy', 'accuracy'])

    return model

def get_model_2rnn(
                  embedding_matrix, cell_size = 80, cell_type_GRU = True,
                  maxlen = 180, max_features = 100000, embed_size = 300,
                  prob_dropout = 0.2, emb_train = False
                 ):
    
    inp_pre = Input(shape=(maxlen, ), name='input_pre')
    inp_post = Input(shape=(maxlen, ), name='input_post')

    ##pre
    x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_pre)
    x1 = SpatialDropout1D(prob_dropout)(x1)
    
    if cell_type_GRU:
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
        x1 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1)
    else :
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
    
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    
    ##post
    x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
    x2 = SpatialDropout1D(prob_dropout)(x2)
    
    if cell_type_GRU:
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
        x2 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2)
    else :
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
    
    avg_pool2 = GlobalAveragePooling1D()(x2)
    max_pool2 = GlobalMaxPooling1D()(x2)
    
    ##merge
    conc = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=[inp_pre, inp_post], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_crossentropy', 'accuracy'])

    return model

def get_model_2rnn_cnn(
                       embedding_matrix, cell_size = 80, cell_type_GRU = True,
                       maxlen = 180, max_features = 100000, embed_size = 300,
                       prob_dropout = 0.2, emb_train = False,
                       filter_size=128, kernel_size = 2, stride = 1
                      ):

    inp_pre = Input(shape=(maxlen, ), name='input_pre')
    inp_post = Input(shape=(maxlen, ), name='input_post')
    
    ##pre
    x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_pre)
    x1 = SpatialDropout1D(prob_dropout)(x1)
    
    if cell_type_GRU:
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
        x1 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1)
    else :
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
    
    x1 = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1)
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    
    ##post
    x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
    x2 = SpatialDropout1D(prob_dropout)(x2)
    
    if cell_type_GRU:
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
        x2 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2)
    else :
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
    
    x2 = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    max_pool2 = GlobalMaxPooling1D()(x2)
    
    ##merge
    conc = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    outp = Dense(6, activation="sigmoid")(conc)
    
    
    model = Model(inputs=[inp_pre, inp_post], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_crossentropy', 'accuracy'])

    return model

def get_model_2rnn_cnn_sp(
                          embedding_matrix, cell_size = 80, cell_type_GRU = True,
                          maxlen = 180, max_features = 100000, embed_size = 300,
                          prob_dropout = 0.2, emb_train = False,
                          filter_size=128, kernel_size = 2, stride = 1
                         ):
    
    inp_pre = Input(shape=(maxlen, ), name='input_pre')
    inp_post = Input(shape=(maxlen, ), name='input_post')
    
    ##pre
    x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_pre)
    x1 = SpatialDropout1D(prob_dropout)(x1)
    
    if cell_type_GRU:
        x1_ = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
        x1 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1_)
    else :
        x1_ = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
        x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1_)
    
    x1_ = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1_)
    avg_pool1_ = GlobalAveragePooling1D()(x1_)
    max_pool1_ = GlobalMaxPooling1D()(x1_)
    
    x1 = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1)
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    
    ##post
    x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
    x2 = SpatialDropout1D(prob_dropout)(x2)
    
    if cell_type_GRU:
        x2_ = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
        x2 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2_)
    else :
        x2_ = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2_)
    
    x2_ = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2_)
    avg_pool2_ = GlobalAveragePooling1D()(x2_)
    max_pool2_ = GlobalMaxPooling1D()(x2_)
    
    x2 = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    max_pool2 = GlobalMaxPooling1D()(x2)
    
    ##merge
    conc = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2, avg_pool1_, max_pool1_, avg_pool2_, max_pool2_])
    outp = Dense(6, activation="sigmoid")(conc)
    
    
    model = Model(inputs=[inp_pre, inp_post], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_crossentropy', 'accuracy'])

    return model

def get_model_dual_2rnn_cnn_sp(
                               embedding_matrix, cell_size = 80, cell_type_GRU = True,
                               maxlen = 180, max_features = 100000, embed_size = 300,
                               prob_dropout = 0.2, emb_train = False,
                               filter_size=128, kernel_size = 2, stride = 1
                              ):
    
    inp_pre = Input(shape=(maxlen, ), name='input_pre')
    inp_post = Input(shape=(maxlen, ), name='input_post')
    
    ##pre
    x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_pre)
    x1g = SpatialDropout1D(prob_dropout)(x1)
    x1l = SpatialDropout1D(prob_dropout)(x1)
    
    x1_g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1g)
    x1g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1_g)
    x1_l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1l)
    x1l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1_l)
    
    x1_g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1_g)
    x1_l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1_l)
    avg_pool1_g = GlobalAveragePooling1D()(x1_g)
    max_pool1_g = GlobalMaxPooling1D()(x1_g)
    avg_pool1_l = GlobalAveragePooling1D()(x1_l)
    max_pool1_l = GlobalMaxPooling1D()(x1_l)
    
    x1g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1g)
    x1l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1l)
    avg_pool1g = GlobalAveragePooling1D()(x1g)
    max_pool1g = GlobalMaxPooling1D()(x1g)
    avg_pool1l = GlobalAveragePooling1D()(x1l)
    max_pool1l = GlobalMaxPooling1D()(x1l)
    
    ##post
    x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
    x2g = SpatialDropout1D(prob_dropout)(x2)
    x2l = SpatialDropout1D(prob_dropout)(x2)
    
    x2_g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2g)
    x2g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2_g)
    x2_l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2l)
    x2l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2_l)
    
    x2_g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2_g)
    x2_l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2_l)
    avg_pool2_g = GlobalAveragePooling1D()(x2_g)
    max_pool2_g = GlobalMaxPooling1D()(x2_g)
    avg_pool2_l = GlobalAveragePooling1D()(x2_l)
    max_pool2_l = GlobalMaxPooling1D()(x2_l)
    
    x2g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2g)
    x2l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2l)
    avg_pool2g = GlobalAveragePooling1D()(x2g)
    max_pool2g = GlobalMaxPooling1D()(x2g)
    avg_pool2l = GlobalAveragePooling1D()(x2l)
    max_pool2l = GlobalMaxPooling1D()(x2l)
    
    ##merge
    conc = concatenate([avg_pool1g, max_pool1g, avg_pool1l, max_pool1l, avg_pool1_g, max_pool1_g, avg_pool1_l, max_pool1_l, 
                        avg_pool2g, max_pool2g, avg_pool2l, max_pool2l, avg_pool2_g, max_pool2_g, avg_pool2_l, max_pool2_l])
    outp = Dense(6, activation="sigmoid")(conc)
    
    
    model = Model(inputs=[inp_pre, inp_post], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_crossentropy', 'accuracy'])

    return model