# coding=utf8
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Activation
from keras.models import Model


def get_embedding_layer(dict_size, embedding_dim):
    embedding_layer = Embedding(dict_size, embedding_dim, weights=[], trainable=False)
    return embedding_layer


def lstm_model(nb_paragraph, nb_sentence, nb_words, dict_size,
               word_embedding_dim, sentence_embedding_dim, paragraph_embedding_dim, document_embedding_dim):
    word_lstm = LSTM(output_dim=sentence_embedding_dim, input_shape=(nb_words, word_embedding_dim))
    sentence_lstm = LSTM(output_dim=paragraph_embedding_dim, input_shape=(nb_sentence, sentence_embedding_dim))
    paragraph_lstm = LSTM(output_dim=document_embedding_dim, input_shape=(nb_paragraph, paragraph_embedding_dim))
    a = TimeDistributed(word_lstm, input_shape=(nb_sentence, nb_words, word_embedding_dim))
    b = TimeDistributed(sentence_lstm, input_shape=(nb_paragraph, nb_sentence, sentence_embedding_dim))
    c = TimeDistributed(a, input_shape=(nb_paragraph, nb_sentence, nb_words, word_embedding_dim))
    d = b(c)
    hierarchical_lstm = paragraph_lstm(d)

    input_layer = Input(shape=(nb_paragraph, nb_sentence, nb_words))
    embedding_layer = get_embedding_layer(dict_size, word_embedding_dim)
    relation_map = Dense(output_dim=document_embedding_dim, input_shape=(document_embedding_dim,))
    output = Activation(relation_map(hierarchical_lstm(embedding_layer(input_layer))))
    return Model(input=input_layer, output=output)


if __name__ == '__main__':
    model = lstm_model(8, 8, 32, 20000, 300, 600, 700, 800)
