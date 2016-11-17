# coding=utf8
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Activation, Reshape
from keras.models import Model


def get_embedding_layer(dict_size, embedding_dim):
    embedding_layer = Embedding(dict_size, embedding_dim, trainable=False)
    return embedding_layer


def lstm_model(nb_paragraph, nb_sentence, nb_words, dict_size,
               word_embedding_dim, sentence_embedding_dim, paragraph_embedding_dim, document_embedding_dim):
    word_lstm = LSTM(output_dim=sentence_embedding_dim, input_shape=(nb_words, word_embedding_dim))
    sentence_lstm = LSTM(output_dim=paragraph_embedding_dim, input_shape=(nb_sentence, sentence_embedding_dim))
    paragraph_lstm = LSTM(output_dim=document_embedding_dim, input_shape=(nb_paragraph, paragraph_embedding_dim))
    relation_layer = Dense(output_dim=document_embedding_dim, input_shape=(document_embedding_dim,))
    # hierarchical_lstm = paragraph_lstm(
    #     TimeDistributed(
    #         # mapping from (nb_sentence, nb_words, word_embedding) to paragraph embedding
    #         sentence_lstm(
    #             TimeDistributed(
    #                 # mapping from (nb_words, word_embedding) to sentence embedding
    #                 word_lstm,
    #                 input_shape=(nb_sentence, nb_words, word_embedding_dim))),
    #         input_shape=(nb_paragraph, nb_sentence, nb_words, word_embedding_dim)))(embedding_layer)
    total_words = nb_words * nb_sentence * nb_paragraph
    input_layer = Input(shape=(total_words,))
    embedding_layer = get_embedding_layer(dict_size, word_embedding_dim)(input_layer)
    first_reshape = Reshape((nb_paragraph * nb_sentence, nb_words, word_embedding_dim))(embedding_layer)
    sentence_embeddings = TimeDistributed(word_lstm)(first_reshape)
    second_reshape = Reshape((nb_paragraph, nb_sentence, sentence_embedding_dim))(sentence_embeddings)
    paragraph_embeddings = TimeDistributed(sentence_lstm)(second_reshape)
    document_embedding = paragraph_lstm(paragraph_embeddings)
    adjusted_score_layer = relation_layer(document_embedding)
    output_layer = Activation(activation=u'softmax')(adjusted_score_layer)
    return Model(input=input_layer, output=output_layer)


if __name__ == '__main__':
    model = lstm_model(8, 8, 32, 20000, 300, 600, 700, 800)

