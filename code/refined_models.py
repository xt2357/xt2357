# coding=utf8
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, \
    Activation, Reshape, Masking, GRU, Lambda, Merge, merge
from keras.models import Model
from keras.regularizers import l2
from keras.models import Sequential
from keras.constraints import unitnorm, maxnorm
from keras import backend as K
import keras
import numpy
import codecs
import preprocessing
import refined_preprocessing
import nlp_utils
import os
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint


def lstm_doc_embedding(nb_sentence, nb_words, dict_size, word_embedding_weights,
                       word_embedding_dim, sentence_embedding_dim, document_embedding_dim):
    word_lstm_model = Sequential()
    word_lstm_model.add(Masking(input_shape=(nb_words, word_embedding_dim), name=u'word_lstm_masking'))
    word_lstm = LSTM(output_dim=sentence_embedding_dim, input_shape=(None, word_embedding_dim),
                     activation=u'tanh', inner_activation=u'hard_sigmoid', name=u'word_lstm')
    word_lstm_model.add(word_lstm)
    sentence_lstm_model = Sequential()
    sentence_lstm_model.add(Masking(input_shape=(nb_sentence, sentence_embedding_dim), name=u'sentence_lstm_masking'))
    sentence_lstm = LSTM(output_dim=document_embedding_dim, input_shape=(None, sentence_embedding_dim),
                         activation=u'tanh', inner_activation=u'hard_sigmoid', name=u'sentence_lstm')
    sentence_lstm_model.add(sentence_lstm)

    total_words = nb_words * nb_sentence
    input_layer = Input(shape=(total_words,))
    embedding_layer = \
        Embedding(dict_size, word_embedding_dim, weights=word_embedding_weights,
                  trainable=False, name=u'word_embedding')(input_layer)
    first_reshape = Reshape((nb_sentence, nb_words, word_embedding_dim))(embedding_layer)
    sentence_embeddings = TimeDistributed(word_lstm_model)(first_reshape)
    document_embedding = sentence_lstm_model(sentence_embeddings)
    model = Model(input=input_layer, output=document_embedding)
    return model


def read_embedding_weights(nyt_word_embedding_path):
    index2embedding = {}
    cnt = 0
    for line in codecs.open(nyt_word_embedding_path, encoding='utf8'):
        # load the most DICT_SIZE frequent words embeddings into embedding layer
        if cnt >= preprocessing.DICT_SIZE:
            break
        cnt += 1
        values = line.split()
        idx = int(values[0])
        coefs = numpy.asarray(values[3:], dtype='float32')
        index2embedding[idx] = coefs
    assert len(index2embedding) == preprocessing.DICT_SIZE, u'word dict size not correct!'
    embedding_weights = numpy.zeros((len(index2embedding), preprocessing.NYT_WORD_EMBEDDING_DIM))
    for i, cof in index2embedding.iteritems():
        embedding_weights[i] = cof
    return [embedding_weights]


def get_lstm_doc_embedding():
    return lstm_doc_embedding(preprocessing.MAX_SENTENCES_IN_DOCUMENT,
                              preprocessing.MAX_WORDS_IN_SENTENCE,
                              preprocessing.DICT_SIZE,
                              read_embedding_weights(
                                  preprocessing.NYT_IGNORE_CASE_WORD_EMBEDDING_PATH),
                              preprocessing.NYT_WORD_EMBEDDING_DIM, 450, 800)


def get_model_by_big_tag(related_big_tag):
    if related_big_tag == u'all_big_tags':
        model = Sequential()
        model.add(get_lstm_doc_embedding())
        model.add(Dense(output_dim=refined_preprocessing.TagManager.BIG_TAG_COUNT,
                        activation=u'softmax', name=related_big_tag))
        model.compile(loss=u'categorical_crossentropy', optimizer=u'rmsprop', metrics=[u'accuracy'])
        return model
    else:
        model = Sequential()
        model.add(get_lstm_doc_embedding())
        related_big_tag_seq = refined_preprocessing.TagManager.BIG_TAG_TO_SEQ[related_big_tag]
        model.add(Dense(output_dim=refined_preprocessing.TagManager.SUBTAG_COUNT[related_big_tag_seq],
                        activation=u'softmax', name=related_big_tag))
        model.compile(loss=u'categorical_crossentropy', optimizer=u'rmsprop', metrics=[u'accuracy'])
        return model


REFINED_MODEL_WEIGHTS_PATH_ROOT = os.path.join(os.path.dirname(__file__), ur'../models/refined/')


def train():
    pass


if __name__ == '__main__':
    big = get_model_by_big_tag(u'all_big_tags')
    big.save_weights(os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'all_big_tags'))
    big = get_model_by_big_tag(u'arts')
    big.load_weights(os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'all_big_tags'), by_name=True)
    pass