# coding=utf8
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Activation, Reshape, Masking
from keras.models import Model
from keras.regularizers import l2
from keras.models import Sequential
from keras import backend as K
import numpy
import preprocessing


def lstm_model(nb_paragraph, nb_sentence, nb_words, dict_size, word_embedding_weights,
               word_embedding_dim, sentence_embedding_dim, paragraph_embedding_dim, document_embedding_dim):
    word_lstm = LSTM(output_dim=sentence_embedding_dim, input_shape=(nb_words, word_embedding_dim),
                     activation=u'sigmoid', inner_activation=u'hard_sigmoid')
    sentence_lstm = LSTM(output_dim=paragraph_embedding_dim, input_shape=(nb_sentence, sentence_embedding_dim),
                         activation=u'sigmoid', inner_activation=u'hard_sigmoid')
    paragraph_lstm = LSTM(output_dim=document_embedding_dim, input_shape=(nb_paragraph, paragraph_embedding_dim),
                          activation=u'sigmoid', inner_activation=u'hard_sigmoid')
    relation_layer = Dense(output_dim=document_embedding_dim, input_shape=(document_embedding_dim,),
                           name=u'relation', bias=False, W_regularizer=l2(0.01))
    total_words = nb_words * nb_sentence * nb_paragraph
    input_layer = Input(shape=(total_words,))
    embedding_layer = \
        Embedding(dict_size, word_embedding_dim, weights=word_embedding_weights, trainable=False)(input_layer)
    first_reshape = Reshape((nb_paragraph * nb_sentence, nb_words, word_embedding_dim))(embedding_layer)
    sentence_embeddings = TimeDistributed(word_lstm)(first_reshape)
    second_reshape = Reshape((nb_paragraph, nb_sentence, sentence_embedding_dim))(sentence_embeddings)
    paragraph_embeddings = TimeDistributed(sentence_lstm)(second_reshape)
    document_embedding = paragraph_lstm(paragraph_embeddings)
    adjusted_score_layer = relation_layer(document_embedding)
    output_layer = Activation(activation=u'softmax')(adjusted_score_layer)
    return Model(input=input_layer, output=output_layer)


def simplified_lstm_model(nb_sentence, nb_words, dict_size, word_embedding_weights,
                          word_embedding_dim, sentence_embedding_dim, document_embedding_dim):
    word_lstm = LSTM(output_dim=sentence_embedding_dim, input_shape=(nb_words, word_embedding_dim),
                     activation=u'sigmoid', inner_activation=u'hard_sigmoid')
    sentence_lstm = LSTM(output_dim=document_embedding_dim, input_shape=(nb_sentence, sentence_embedding_dim),
                         activation=u'sigmoid', inner_activation=u'hard_sigmoid')
    relation_layer = Dense(output_dim=document_embedding_dim, input_shape=(document_embedding_dim,),
                           name=u'relation', bias=False, W_regularizer=l2(0.01))
    total_words = nb_words * nb_sentence
    input_layer = Input(shape=(total_words,))
    embedding_layer = \
        Embedding(dict_size, word_embedding_dim, weights=word_embedding_weights, trainable=False)(input_layer)
    first_reshape = Reshape((nb_sentence, nb_words, word_embedding_dim))(embedding_layer)
    sentence_embeddings = TimeDistributed(word_lstm)(first_reshape)
    document_embedding = sentence_lstm(sentence_embeddings)
    adjusted_score_layer = relation_layer(document_embedding)
    output_layer = Activation(activation=u'softmax')(adjusted_score_layer)
    return Model(input=input_layer, output=output_layer)


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
                           name=u'relation', bias=False, W_regularizer=l2(0.01))
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

    model = Model(input=input_layer, output=output_layer)
    model.compile(loss=masked_simplified_lstm_loss, optimizer='rmsprop')
    return model


def test_mask_model():
    inner_model = Sequential()
    inner_model.add(Masking(input_shape=(3, 4)))
    word_lstm = LSTM(output_dim=8, input_shape=(None, 4),
                     activation=u'sigmoid', inner_activation=u'hard_sigmoid')
    inner_model.add(word_lstm)
    input_layer = Input(shape=(6, 4))
    first_reshape = Reshape((2, 3, 4))(input_layer)
    sentence_embeddings = TimeDistributed(inner_model)(first_reshape)
    return Model(input=input_layer, output=sentence_embeddings)


def test_mask():
    my_model = test_mask_model()
    print (my_model.predict(
        [numpy.array([[[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]])]))
    print (u'hello')
    print (my_model.predict(
        [numpy.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [6, 6, 6, 6], [3, 3, 3, 3], [0, 0, 0, 0]]])]))


def main():
    model = masked_simplified_lstm(preprocessing.MAX_SENTENCES_IN_DOCUMENT, preprocessing.MAX_WORDS_IN_SENTENCE,
                                   preprocessing.DICT_SIZE, None, 300, 600, preprocessing.TAG_DICT_SIZE)


if __name__ == '__main__':
    main()
    pass
