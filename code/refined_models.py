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
REFINED_DATA_PATH_ROOT = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/')


def get_big_tag_model_save_path(big_tag):
    return os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'%s.h5' % big_tag)


def train_big_tag_model(big_tag, x_train, y_train, x_eval, y_eval):
    model_weights_save_path = get_big_tag_model_save_path(big_tag)
    check_point_save_path = os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'%s {epoch:02d}.h5' % big_tag)
    model = get_model_by_big_tag(big_tag)
    # load pre-trained weights
    if big_tag != u'all_big_tags':
        model.load_weights(os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'all_big_tags.h5'), by_name=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    check_point = ModelCheckpoint(check_point_save_path, save_weights_only=True)
    model.fit(x_train, y_train, validation_data=(x_eval, y_eval),
              batch_size=64, nb_epoch=4, callbacks=[early_stopping, check_point])
    model.save_weights(model_weights_save_path)
    print (u'model train done, weights saved to %s' % model_weights_save_path)


def train():
    x_train, x_eval = \
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_TRAIN_SP),\
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_EVAL_SP)
    y_train, y_eval = \
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_Y_TRAIN_SP),\
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_Y_EVAL_SP)

    all_big_tag_x_train, all_big_tag_y_train = refined_preprocessing.filter_x_y(x_train, y_train, u'all_big_tags')
    all_big_tag_x_eval, all_big_tag_y_eval = refined_preprocessing.filter_x_y(x_eval, y_eval, u'all_big_tags')
    train_big_tag_model(u'all_big_tags', all_big_tag_x_train, all_big_tag_y_train,
                        all_big_tag_x_eval, all_big_tag_y_eval)
    for big_tag, big_tag_seq in refined_preprocessing.TagManager.BIG_TAG_TO_SEQ.items():
        if refined_preprocessing.TagManager.SUBTAG_COUNT[big_tag_seq] == 0:
            continue
        cur_x_train, cur_y_train = refined_preprocessing.filter_x_y(x_train, y_train, big_tag)
        cur_x_eval, cur_y_eval = refined_preprocessing.filter_x_y(x_eval, y_eval, big_tag)
        train_big_tag_model(big_tag, cur_x_train, cur_y_train, cur_x_eval, cur_y_eval)


def load_the_whole_model():
    models = {}
    for big_tag, big_tag_seq in refined_preprocessing.TagManager.BIG_TAG_TO_SEQ.items():
        models[big_tag] = get_model_by_big_tag(big_tag)
        models[big_tag].load_weights(get_big_tag_model_save_path(big_tag))
    return models


def predict_based_on_a_star(models, x):
    first_predict = models[u'all_big_tags'].predict(input)



if __name__ == '__main__':
    if sys.argv[1] == u'train':
        train()
