#===============keras ==============
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, add
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.layers import K, Activation
from keras.engine import Layer
from keras import initializers, regularizers, constraints


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    
    
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

def get_model_rnn_caps(
                      embedding_matrix, cell_size = 80, cell_type_GRU = True,
                      maxlen = 180, max_features = 100000, embed_size = 300,
                      prob_dropout = 0.2, emb_train = False,
                      Routings = 5, Num_capsule = 10, Dim_capsule = 16
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
    
    capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x1)
    capsule1 = Flatten()(capsule1)
    
    
    ##post
    x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
    x2 = SpatialDropout1D(prob_dropout)(x2)
    
    if cell_type_GRU:
        x2 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2)
    else :
        x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
    
    capsule2 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x2)
    capsule2 = Flatten()(capsule2)
    
    
    ##merge
    conc = concatenate([capsule1, capsule2])
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

# def get_model_2rnn_cnn_sp(
#                           embedding_matrix, cell_size = 80, cell_type_GRU = True,
#                           maxlen = 180, max_features = 100000, embed_size = 300,
#                           prob_dropout = 0.2, emb_train = False,
#                           filter_size=128, kernel_size = 2, stride = 1
#                          ):
    
#     inp_pre = Input(shape=(maxlen, ), name='input_pre')
#     inp_post = Input(shape=(maxlen, ), name='input_post')
    
#     ##pre
#     x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_pre)
#     x1 = SpatialDropout1D(prob_dropout)(x1)
    
#     if cell_type_GRU:
#         x1_ = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
#         x1 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1_)
#     else :
#         x1_ = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1)
#         x1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1_)
    
#     x1_ = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1_)
#     avg_pool1_ = GlobalAveragePooling1D()(x1_)
#     max_pool1_ = GlobalMaxPooling1D()(x1_)
    
#     x1 = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1)
#     avg_pool1 = GlobalAveragePooling1D()(x1)
#     max_pool1 = GlobalMaxPooling1D()(x1)
    
#     ##post
#     x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
#     x2 = SpatialDropout1D(prob_dropout)(x2)
    
#     if cell_type_GRU:
#         x2_ = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
#         x2 = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2_)
#     else :
#         x2_ = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2)
#         x2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2_)
    
#     x2_ = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2_)
#     avg_pool2_ = GlobalAveragePooling1D()(x2_)
#     max_pool2_ = GlobalMaxPooling1D()(x2_)
    
#     x2 = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2)
#     avg_pool2 = GlobalAveragePooling1D()(x2)
#     max_pool2 = GlobalMaxPooling1D()(x2)
    
#     ##merge
#     conc = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2, avg_pool1_, max_pool1_, avg_pool2_, max_pool2_])
#     outp = Dense(6, activation="sigmoid")(conc)
    
    
#     model = Model(inputs=[inp_pre, inp_post], outputs=outp)
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['binary_crossentropy', 'accuracy'])

#     return model

# def get_model_dual_2rnn_cnn_sp(
#                                embedding_matrix, cell_size = 80, cell_type_GRU = True,
#                                maxlen = 180, max_features = 100000, embed_size = 300,
#                                prob_dropout = 0.2, emb_train = False,
#                                filter_size=128, kernel_size = 2, stride = 1
#                               ):
    
#     inp_pre = Input(shape=(maxlen, ), name='input_pre')
#     inp_post = Input(shape=(maxlen, ), name='input_post')
    
#     ##pre
#     x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_pre)
#     x1g = SpatialDropout1D(prob_dropout)(x1)
#     x1l = SpatialDropout1D(prob_dropout)(x1)
    
#     x1_g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1g)
#     x1g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1_g)
#     x1_l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1l)
#     x1l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1_l)
    
#     x1_g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1_g)
#     x1_l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1_l)
#     avg_pool1_g = GlobalAveragePooling1D()(x1_g)
#     max_pool1_g = GlobalMaxPooling1D()(x1_g)
#     avg_pool1_l = GlobalAveragePooling1D()(x1_l)
#     max_pool1_l = GlobalMaxPooling1D()(x1_l)
    
#     x1g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1g)
#     x1l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1l)
#     avg_pool1g = GlobalAveragePooling1D()(x1g)
#     max_pool1g = GlobalMaxPooling1D()(x1g)
#     avg_pool1l = GlobalAveragePooling1D()(x1l)
#     max_pool1l = GlobalMaxPooling1D()(x1l)
    
#     ##post
#     x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
#     x2g = SpatialDropout1D(prob_dropout)(x2)
#     x2l = SpatialDropout1D(prob_dropout)(x2)
    
#     x2_g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2g)
#     x2g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2_g)
#     x2_l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2l)
#     x2l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2_l)
    
#     x2_g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2_g)
#     x2_l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2_l)
#     avg_pool2_g = GlobalAveragePooling1D()(x2_g)
#     max_pool2_g = GlobalMaxPooling1D()(x2_g)
#     avg_pool2_l = GlobalAveragePooling1D()(x2_l)
#     max_pool2_l = GlobalMaxPooling1D()(x2_l)
    
#     x2g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2g)
#     x2l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2l)
#     avg_pool2g = GlobalAveragePooling1D()(x2g)
#     max_pool2g = GlobalMaxPooling1D()(x2g)
#     avg_pool2l = GlobalAveragePooling1D()(x2l)
#     max_pool2l = GlobalMaxPooling1D()(x2l)
    
#     ##merge
#     conc = concatenate([avg_pool1g, max_pool1g, avg_pool1l, max_pool1l, avg_pool1_g, max_pool1_g, avg_pool1_l, max_pool1_l, 
#                         avg_pool2g, max_pool2g, avg_pool2l, max_pool2l, avg_pool2_g, max_pool2_g, avg_pool2_l, max_pool2_l])
#     outp = Dense(6, activation="sigmoid")(conc)
    
    
#     model = Model(inputs=[inp_pre, inp_post], outputs=outp)
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['binary_crossentropy', 'accuracy'])

#     return model

# def get_model_dual_2rnn_cnn_sp_drop(
#                                embedding_matrix, cell_size = 80, cell_type_GRU = True,
#                                maxlen = 180, max_features = 100000, embed_size = 300,
#                                prob_dropout = 0.2, emb_train = False,
#                                filter_size=128, kernel_size = 2, stride = 1
#                               ):
    
#     inp_pre = Input(shape=(maxlen, ), name='input_pre')
#     inp_post = Input(shape=(maxlen, ), name='input_post')
    
#     ##pre
#     x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_pre)
#     x1g = SpatialDropout1D(prob_dropout)(x1)
#     x1l = SpatialDropout1D(prob_dropout)(x1)
    
#     x1_g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1g)
#     x1g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x1_g)
#     x1_l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1l)
#     x1l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x1_l)
    
#     x1_g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1_g)
#     x1_l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1_l)
#     avg_pool1_g = GlobalAveragePooling1D()(x1_g)
#     max_pool1_g = GlobalMaxPooling1D()(x1_g)
#     avg_pool1_l = GlobalAveragePooling1D()(x1_l)
#     max_pool1_l = GlobalMaxPooling1D()(x1_l)
    
#     x1g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1g)
#     x1l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x1l)
#     avg_pool1g = GlobalAveragePooling1D()(x1g)
#     max_pool1g = GlobalMaxPooling1D()(x1g)
#     avg_pool1l = GlobalAveragePooling1D()(x1l)
#     max_pool1l = GlobalMaxPooling1D()(x1l)
    
#     ##post
#     x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = emb_train)(inp_post)
#     x2g = SpatialDropout1D(prob_dropout)(x2)
#     x2l = SpatialDropout1D(prob_dropout)(x2)
    
#     x2_g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2g)
#     x2g = Bidirectional(CuDNNGRU(cell_size, return_sequences=True))(x2_g)
#     x2_l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2l)
#     x2l = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True))(x2_l)
    
#     x2_g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2_g)
#     x2_l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2_l)
#     avg_pool2_g = GlobalAveragePooling1D()(x2_g)
#     max_pool2_g = GlobalMaxPooling1D()(x2_g)
#     avg_pool2_l = GlobalAveragePooling1D()(x2_l)
#     max_pool2_l = GlobalMaxPooling1D()(x2_l)
    
#     x2g = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2g)
#     x2l = Conv1D(filter_size, kernel_size = kernel_size, strides=stride, padding = "valid", kernel_initializer = "he_uniform")(x2l)
#     avg_pool2g = GlobalAveragePooling1D()(x2g)
#     max_pool2g = GlobalMaxPooling1D()(x2g)
#     avg_pool2l = GlobalAveragePooling1D()(x2l)
#     max_pool2l = GlobalMaxPooling1D()(x2l)
    
#     ##merge
#     conc = concatenate([avg_pool1g, max_pool1g, avg_pool1l, max_pool1l, avg_pool1_g, max_pool1_g, avg_pool1_l, max_pool1_l, 
#                         avg_pool2g, max_pool2g, avg_pool2l, max_pool2l, avg_pool2_g, max_pool2_g, avg_pool2_l, max_pool2_l])
#     conc = SpatialDropout1D(prob_dropout)(conc)
#     outp = Dense(6, activation="sigmoid")(conc)
    
    
#     model = Model(inputs=[inp_pre, inp_post], outputs=outp)
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['binary_crossentropy', 'accuracy'])

#     return model


# def get_model_dpcnn(
#                     embedding_matrix, cell_size = 80, cell_type_GRU = True,
#                     maxlen = 180, max_features = 100000, embed_size = 300,
#                     prob_dropout = 0.2, emb_train = False,
#                     filter_nr=128, filter_size = 2, stride = 1, 
#                     max_pool_size = 3, max_pool_strides = 2, dense_nr = 256,
#                     spatial_dropout = 0.2, dense_dropout = 0.5,
#                     conv_kern_reg = regularizers.l2(0.00001), conv_bias_reg = regularizers.l2(0.00001)
#                     ):

#     comment = Input(shape=(maxlen,))
#     emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=emb_train)(comment)
#     emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

#     block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
#     block1 = BatchNormalization()(block1)
#     block1 = PReLU()(block1)
#     block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
#     block1 = BatchNormalization()(block1)
#     block1 = PReLU()(block1)

#     #we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
#     #if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
#     resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
#     resize_emb = PReLU()(resize_emb)

#     block1_output = add([block1, resize_emb])
#     block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

#     block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
#     block2 = BatchNormalization()(block2)
#     block2 = PReLU()(block2)
#     block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
#     block2 = BatchNormalization()(block2)
#     block2 = PReLU()(block2)

#     block2_output = add([block2, block1_output])
#     block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

#     block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
#     block3 = BatchNormalization()(block3)
#     block3 = PReLU()(block3)
#     block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
#     block3 = BatchNormalization()(block3)
#     block3 = PReLU()(block3)

#     block3_output = add([block3, block2_output])
#     block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

#     block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
#     block4 = BatchNormalization()(block4)
#     block4 = PReLU()(block4)
#     block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
#     block4 = BatchNormalization()(block4)
#     block4 = PReLU()(block4)

#     block4_output = add([block4, block3_output])
#     block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

#     block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
#     block5 = BatchNormalization()(block5)
#     block5 = PReLU()(block5)
#     block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
#     block5 = BatchNormalization()(block5)
#     block5 = PReLU()(block5)

#     block5_output = add([block5, block4_output])
#     block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

#     block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
#     block6 = BatchNormalization()(block6)
#     block6 = PReLU()(block6)
#     block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
#     block6 = BatchNormalization()(block6)
#     block6 = PReLU()(block6)

#     block6_output = add([block6, block5_output])
#     block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

#     block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
#     block7 = BatchNormalization()(block7)
#     block7 = PReLU()(block7)
#     block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
#                 kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
#     block7 = BatchNormalization()(block7)
#     block7 = PReLU()(block7)

#     block7_output = add([block7, block6_output])
#     output = GlobalMaxPooling1D()(block7_output)

#     output = Dense(dense_nr, activation='linear')(output)
#     output = BatchNormalization()(output)
#     output = PReLU()(output)
#     output = Dropout(dense_dropout)(output)
#     output = Dense(6, activation='sigmoid')(output)

#     model = Model(comment, output)


#     model.compile(loss='binary_crossentropy', 
#                 optimizer='adam',
#                 metrics=['accuracy'])
    
#     return model