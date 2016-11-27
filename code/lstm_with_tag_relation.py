# coding=utf8
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Activation, Reshape, Masking
from keras.models import Model
from keras.regularizers import l2
from keras.models import Sequential
from keras.constraints import unitnorm, maxnorm
from keras import backend as K
import numpy
import codecs
import preprocessing
from keras.callbacks import EarlyStopping


def masked_simplified_lstm(nb_sentence, nb_words, dict_size, word_embedding_weights,
                           word_embedding_dim, sentence_embedding_dim, document_embedding_dim):
    word_lstm_model = Sequential()
    word_lstm_model.add(Masking(input_shape=(nb_words, word_embedding_dim)))
    word_lstm = LSTM(output_dim=sentence_embedding_dim, input_shape=(None, word_embedding_dim),
                     activation=u'sigmoid', inner_activation=u'hard_sigmoid')
    word_lstm_model.add(word_lstm)
    sentence_lstm_model = Sequential()
    sentence_lstm_model.add(Masking(input_shape=(nb_sentence, sentence_embedding_dim)))
    sentence_lstm = LSTM(output_dim=document_embedding_dim, input_shape=(None, sentence_embedding_dim),
                         activation=u'sigmoid', inner_activation=u'hard_sigmoid')
    sentence_lstm_model.add(sentence_lstm)
    relation_layer = Dense(output_dim=document_embedding_dim, input_shape=(document_embedding_dim,),
                           name=u'relation', bias=False, W_regularizer=l2(0.01), W_constraint=unitnorm())
    total_words = nb_words * nb_sentence
    input_layer = Input(shape=(total_words,))
    embedding_layer = \
        Embedding(dict_size, word_embedding_dim, weights=word_embedding_weights, trainable=False)(input_layer)
    first_reshape = Reshape((nb_sentence, nb_words, word_embedding_dim))(embedding_layer)
    sentence_embeddings = TimeDistributed(word_lstm_model)(first_reshape)
    document_embedding = sentence_lstm_model(sentence_embeddings)
    adjusted_score_layer = relation_layer(document_embedding)
    output_layer = Activation(activation=u'softmax')(adjusted_score_layer)

    def masked_simplified_lstm_loss(y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true) - K.sum(y_true * relation_layer.call(y_true), axis=-1)

    def masked_simplified_lstm_loss_without_relation(y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true)

    model = Model(input=input_layer, output=output_layer)
    model.compile(loss=masked_simplified_lstm_loss_without_relation, optimizer='rmsprop')
    return model


def new_model():
    return masked_simplified_lstm(preprocessing.MAX_SENTENCES_IN_DOCUMENT,
                                  preprocessing.MAX_WORDS_IN_SENTENCE,
                                  preprocessing.DICT_SIZE,
                                  get_embedding_weights(
                                      preprocessing.NYT_IGNORE_CASE_WORD_EMBEDDING_PATH),
                                  preprocessing.NYT_WORD_EMBEDDING_DIM, 400, preprocessing.TAG_DICT_SIZE)


def get_embedding_weights(nyt_word_embedding_path):
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


MODEL_WEIGHTS_PATH = u'../models/model_weights.h5'
HISTORY_PATH = u'../models/train_history.txt'


def train():
    model = new_model()
    x_train, y_train = \
        preprocessing.read_x(preprocessing.X_TRAIN_PATH), preprocessing.read_y(preprocessing.Y_TRAIN_PATH)
    print (u'train data loaded')
    x_eval, y_eval = \
        preprocessing.read_x(preprocessing.X_EVAL_PATH), preprocessing.read_y(preprocessing.Y_EVAL_PATH)
    print (u'eval data loaded')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(x_train, y_train, callbacks=[early_stopping], validation_data=(x_eval, y_eval),
                        nb_epoch=4, batch_size=64)
    model.save_weights(MODEL_WEIGHTS_PATH)
    print (u'model saved to %s' % MODEL_WEIGHTS_PATH)
    print (history.history)
    with codecs.open(HISTORY_PATH, 'w', 'utf8') as history_output:
        history_output.write(unicode(history.history))


def main():
    model = new_model()
    # model.save_weights(u'../models/model_weights.h5')
    model.load_weights(u'../models/model_weights.h5')


if __name__ == '__main__':
    main()
    pass
