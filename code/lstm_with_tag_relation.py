# coding=utf8
from keras.layers import Input, Embedding


def get_embedding_layer(dict_size, embedding_dim):
    embedding_layer = Embedding(dict_size, embedding_dim, weights=[], trainable=False)
    return embedding_layer


def lstm_model(nb_paragraph, nb_sentence, nb_words):
    inputs = Input(shape=(None, None, None))