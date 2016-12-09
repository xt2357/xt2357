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
import nlp_utils
import os
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint


def masked_simplified_lstm(nb_sentence, nb_words, dict_size, word_embedding_weights,
                           word_embedding_dim, sentence_embedding_dim, document_embedding_dim, nb_tags):
    word_lstm_model = Sequential()
    word_lstm_model.add(Masking(input_shape=(nb_words, word_embedding_dim)))
    word_lstm = LSTM(output_dim=sentence_embedding_dim, input_shape=(None, word_embedding_dim),
                    activation=u'tanh', inner_activation=u'hard_sigmoid')
    word_lstm_model.add(word_lstm)
    sentence_lstm_model = Sequential()
    sentence_lstm_model.add(Masking(input_shape=(nb_sentence, sentence_embedding_dim)))
    sentence_lstm = LSTM(output_dim=document_embedding_dim, input_shape=(None, sentence_embedding_dim),
                        activation=u'tanh', inner_activation=u'hard_sigmoid')
    sentence_lstm_model.add(sentence_lstm)
    relation_layer = Dense(output_dim=nb_tags, input_shape=(nb_tags,),
                           name=u'relation', bias=False, W_regularizer=l2(0.01), W_constraint=unitnorm(axis=0))
    total_words = nb_words * nb_sentence
    input_layer = Input(shape=(total_words,))
    embedding_layer = \
        Embedding(dict_size, word_embedding_dim, weights=word_embedding_weights, trainable=False)(input_layer)
    first_reshape = Reshape((nb_sentence, nb_words, word_embedding_dim))(embedding_layer)
    sentence_embeddings = TimeDistributed(word_lstm_model)(first_reshape)
    document_embedding = sentence_lstm_model(sentence_embeddings)
    dense_layer = Dense(output_dim=nb_tags, input_shape=(document_embedding_dim,), activation=u'tanh',
                        W_regularizer=l2(0.01))(document_embedding)
    adjusted_score_layer = relation_layer(dense_layer)
    output_layer = Activation(activation=u'softmax')(adjusted_score_layer)

    def masked_simplified_lstm_loss(y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true) - K.sum(y_true * relation_layer.call(y_true), axis=-1)

    def masked_simplified_lstm_loss_cross_entropy(y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true) + \
               K.categorical_crossentropy(y_true, relation_layer.call(y_true))

    def masked_simplified_lstm_loss_without_relation(y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true)

    model = Model(input=input_layer, output=output_layer)
    model.compile(loss=masked_simplified_lstm_loss, optimizer='rmsprop')
    return model


def gru_with_attention(nb_sentence, nb_words, dict_size, word_embedding_weights,
                       word_embedding_dim, sentence_embedding_dim, document_embedding_dim, nb_tags):
    word_lstm_input = Input(shape=(preprocessing.MAX_WORDS_IN_SENTENCE, word_embedding_dim))
    word_lstm_output = GRU(output_dim=sentence_embedding_dim, return_sequences=True,
                           input_shape=(preprocessing.MAX_WORDS_IN_SENTENCE, word_embedding_dim),
                           activation=u'tanh', inner_activation=u'hard_sigmoid')(word_lstm_input)

    def get_last(word_lstm_output_seq):
        return K.permute_dimensions(word_lstm_output_seq, (1, 0, 2))[-1]

    for_get_last = Lambda(get_last, output_shape=(sentence_embedding_dim,))(word_lstm_output)
    sentence_context = \
        Dense(output_dim=preprocessing.MAX_WORDS_IN_SENTENCE, activation=u'tanh')(for_get_last)
    weights = Dense(output_dim=preprocessing.MAX_WORDS_IN_SENTENCE, activation=u'softmax')(sentence_context)
    final_output = merge([weights, word_lstm_output], mode=u'dot', dot_axes=1)
    word_lstm_model = Model(input=[word_lstm_input], output=[final_output])

    sentence_lstm_model = Sequential()
    sentence_lstm = GRU(output_dim=document_embedding_dim,
                        input_shape=(preprocessing.MAX_SENTENCES_IN_DOCUMENT, sentence_embedding_dim),
                        activation=u'tanh', inner_activation=u'hard_sigmoid')
    sentence_lstm_model.add(sentence_lstm)
    relation_layer = Dense(output_dim=nb_tags, input_shape=(nb_tags,),
                           name=u'relation', bias=False, W_regularizer=l2(0.01), W_constraint=unitnorm(axis=0))
    total_words = nb_words * nb_sentence
    input_layer = Input(shape=(total_words,))
    embedding_layer = \
        Embedding(dict_size, word_embedding_dim, weights=word_embedding_weights, trainable=False)(input_layer)
    first_reshape = Reshape((nb_sentence, nb_words, word_embedding_dim))(embedding_layer)
    sentence_embeddings = TimeDistributed(word_lstm_model)(first_reshape)
    document_embedding = sentence_lstm_model(sentence_embeddings)
    dense_layer = Dense(output_dim=nb_tags, input_shape=(document_embedding_dim,), activation=u'tanh',
                        W_regularizer=l2(0.01))(document_embedding)
    adjusted_score_layer = relation_layer(dense_layer)
    output_layer = Activation(activation=u'softmax')(adjusted_score_layer)

    def masked_simplified_lstm_loss(y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true) - K.sum(y_true * relation_layer.call(y_true), axis=-1)

    def masked_simplified_lstm_loss_cross_entropy(y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true) + \
               K.categorical_crossentropy(y_true, relation_layer.call(y_true))

    def masked_simplified_lstm_loss_without_relation(y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true)

    model = Model(input=input_layer, output=output_layer)
    model.compile(loss=masked_simplified_lstm_loss, optimizer='rmsprop')
    return model


def new_model(tag_count=preprocessing.MEANINGFUL_TAG_SIZE):
    return masked_simplified_lstm(preprocessing.MAX_SENTENCES_IN_DOCUMENT,
                                  preprocessing.MAX_WORDS_IN_SENTENCE,
                                  preprocessing.DICT_SIZE,
                                  read_embedding_weights(
                                      preprocessing.NYT_IGNORE_CASE_WORD_EMBEDDING_PATH),
                                  preprocessing.NYT_WORD_EMBEDDING_DIM, 450, 800, tag_count)


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


MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), u'../models/model_weights.h5')
HISTORY_PATH = os.path.join(os.path.dirname(__file__), u'../models/train_history.txt')


def get_sample_weights_template():
    v = numpy.zeros(preprocessing.MEANINGFUL_TAG_SIZE)
    for line in open(preprocessing.NYT_TAG_DICT_PATH):
        idx = int(line.split()[0])
        if idx < preprocessing.MEANINGFUL_TAG_SIZE:
            v[idx] = 1.0 / int(line.split()[2])
    return v


MODEL_PATH = os.path.join(os.path.dirname(__file__), u"../models/model-{epoch:02d}.h5")


def train_on_refined_data():
    import refined_preprocessing
    model = new_model(refined_preprocessing.TagManager.REFINED_TAG_DICT_SIZE)
    x_train, y_train = \
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_TRAIN_SP),\
        refined_preprocessing.read_refined_y(refined_preprocessing.REFINED_Y_TRAIN_SP, return_idx=True)
    print (u'train data loaded')
    x_eval, y_eval = \
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_EVAL_SP), \
        refined_preprocessing.read_refined_y(refined_preprocessing.REFINED_Y_EVAL_SP, return_idx=True)
    print (u'eval data loaded')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    check_point = ModelCheckpoint(MODEL_PATH, save_weights_only=True)
    history = model.fit(x_train, y_train, callbacks=[early_stopping, check_point], validation_data=(x_eval, y_eval),
                        nb_epoch=4, batch_size=64, sample_weight=None)
    model.save_weights(MODEL_WEIGHTS_PATH)
    print (u'model saved to %s' % MODEL_WEIGHTS_PATH)
    print (history.history)
    with codecs.open(HISTORY_PATH, 'w', 'utf8') as history_output:
        history_output.write(unicode(history.history))
    lsq(MODEL_WEIGHTS_PATH, x_train, y_train, on_refined_data=True)


def train():
    model = new_model()
    x_train, y_train = \
        preprocessing.read_x(preprocessing.X_TRAIN_IGNORE_STOP_PATH), \
        preprocessing.read_y(preprocessing.Y_TRAIN_IGNORE_STOP_PATH)
    # sample_weights = numpy.zeros(len(y_train))
    # template = get_sample_weights_template()
    # for i in range(len(y_train)):
    #     sample_weights[i] = numpy.sum(y_train[i] * template)
    print (u'train data loaded')
    x_eval, y_eval = \
        preprocessing.read_x(preprocessing.X_EVAL_IGNORE_STOP_PATH), \
        preprocessing.read_y(preprocessing.Y_EVAL_IGNORE_STOP_PATH)
    print (u'eval data loaded')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    check_point = ModelCheckpoint(MODEL_PATH, save_weights_only=True)
    history = model.fit(x_train, y_train, callbacks=[early_stopping, check_point], validation_data=(x_eval, y_eval),
                        nb_epoch=4, batch_size=64, sample_weight=None)
    model.save_weights(MODEL_WEIGHTS_PATH)
    print (u'model saved to %s' % MODEL_WEIGHTS_PATH)
    print (history.history)
    with codecs.open(HISTORY_PATH, 'w', 'utf8') as history_output:
        history_output.write(unicode(history.history))
    lsq(MODEL_WEIGHTS_PATH, x_train, y_train)


def lsq(trained_model_path, x_train=None, y_train=None, sample_size=None, on_refined_data=False):
    if x_train is None:
        if on_refined_data:
            import refined_preprocessing
            x_train, y_train = \
                refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_TRAIN_SP, sample_size), \
                refined_preprocessing.read_refined_y(refined_preprocessing.REFINED_Y_TRAIN_SP, sample_size, return_idx=True)
        else:
            x_train, y_train = \
                preprocessing.read_x(preprocessing.X_TRAIN_IGNORE_STOP_PATH, sample_size), \
                preprocessing.read_y(preprocessing.Y_TRAIN_IGNORE_STOP_PATH, sample_size)
        print (u'train data loaded')
    from scipy.optimize import leastsq
    from numpy.linalg import lstsq
    model = new_model()
    print (u'lsq loading model %s' % trained_model_path)
    if trained_model_path.endswith(u'hdf5'):
        model = keras.models.load_model(trained_model_path)
    else:
        model.load_weights(trained_model_path)
    print (u'model %s loaded, prediction start..' % trained_model_path)
    y_pred = model.predict(x_train)
    sorted_y_pred = numpy.sort(y_pred)
    sorted_y_pred_args = numpy.argsort(y_pred)
    print (u'prediction and sort done..generate optimal threshold vector..')
    optimal_threshold_v = numpy.zeros(len(x_train))
    nb_tags = len(y_train[0])
    aver_precision, aver_recall = 0.0, 0.0
    for i in range(len(x_train)):
        true_tags = derive_tag_indices_from_y(y_train[i], is_y_true=True)
        precision, recall, pos = 0.0, 0.0, -1
        correct_tags_predicted = 0.0
        nb_misclassified = sys.maxint
        # searching optimal threshold
        for idx in range(nb_tags - 1, -1, -1):
            # for speeding up
            if nb_tags - idx > 10:
                break
            if sorted_y_pred_args[i][idx] not in true_tags:
                continue
            correct_tags_predicted += 1
            this_precision, this_recall = \
                correct_tags_predicted / (nb_tags - idx), correct_tags_predicted / len(true_tags)
            if (nb_tags - idx - correct_tags_predicted) + (len(true_tags) - correct_tags_predicted) < nb_misclassified:
                precision, recall, pos = this_precision, this_recall, idx
                nb_misclassified = (nb_tags - idx - correct_tags_predicted) + (len(true_tags) - correct_tags_predicted)
        aver_precision, aver_recall = (aver_precision * i + precision) / (i + 1), (aver_recall * i + recall) / (i + 1)
        optimal_threshold_v[i] = \
            (sorted_y_pred[i][pos] + (sorted_y_pred[i][pos - 1] if pos != 0 else sorted_y_pred[i][pos])) / 2.0
    print (u'optimal threshold retrieve done, average precision: %lf, average recall: %lf..'
           u'starting least square..' % (aver_precision, aver_recall))
    amend_y_pred = numpy.column_stack((y_pred, numpy.ones(len(y_pred))))

    def lsq_func(threshold_lsq_coefficient):
        return amend_y_pred.dot(threshold_lsq_coefficient) - optimal_threshold_v

    #### ans = leastsq(lsq_func, numpy.random.rand(nb_tags + 1))
    ans = lstsq(amend_y_pred, optimal_threshold_v)
    print (u'square loss: %lf' % numpy.sum(lsq_func(ans[0]) ** 2))
    numpy.savetxt(THRESHOLD_LSQ_COEFFICIENT_PATH, ans[0])
    print (u'least square done..')
    # lr = get_lr_model(len(y_pred[0]))
    # early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # import refined_preprocessing
    # datum = refined_preprocessing.randomly_split_data_in_memory(0.1, y_pred, optimal_threshold_v)
    # lr.fit(datum[u'x_train'], datum[u'y_train'], validation_data=(datum[u'x_eval'], datum[u'y_eval']),
    #        batch_size=128, nb_epoch=6, callbacks=[early_stopping])
    # lr.save_weights(LR_MODEL_WEIGHTS_PATH)
    # print (u'lr model weights saved to %s' % LR_MODEL_WEIGHTS_PATH)


def get_lr_model(input_dim):
    lr = Sequential()
    lr.add(Dense(output_dim=1, input_shape=(input_dim,), activation='sigmoid', W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
    lr.compile(optimizer='sgd', loss='mean_squared_error')
    return lr

LR_MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), u"../models/lr_weights.h5")


def text_predict(trained_model, text, cutoff=10):
    # replace all the \n to make the whole text a single paragraph
    structure = nlp_utils.split_into_paragraph_sentence_token(text.replace(u'\n', u''))
    assert len(structure) != 0 % u'text empty!(ignore stopwords)'
    sentences = structure[0]
    input_v = preprocessing.padding_document(sentences)
    output_v = trained_model.predict([input_v])[0]
    numpy.sort(output_v)
    idx = len(output_v) - 1
    tags = []
    while True:
        if idx < 0 or len(output_v) - idx > cutoff:
            break
        tags.append((idx, output_v[idx]))
        idx -= 1
    return [(preprocessing.TAG_IDX_TO_NAME[idx], confidence) for idx, confidence in tags]


THRESHOLD_LSQ_COEFFICIENT_PATH = os.path.join(os.path.dirname(__file__), u'../models/threshold_lsq_coefficient.txt')
THRESHOLD_LSQ_COEFFICIENT = None
LR_MODEL = None


# using adaptive threshold mechanism: threshold = sorted_nn_output_v dot threshold_lsq_coefficient
def read_threshold_lsq_coefficient():
    global THRESHOLD_LSQ_COEFFICIENT, LR_MODEL
    # LR_MODEL = get_lr_model(88)
    # LR_MODEL.load_weights(LR_MODEL_WEIGHTS_PATH)
    # print (u'lr model read')
    if THRESHOLD_LSQ_COEFFICIENT:
        return
    if os.path.exists(THRESHOLD_LSQ_COEFFICIENT_PATH):
        THRESHOLD_LSQ_COEFFICIENT = numpy.loadtxt(open(THRESHOLD_LSQ_COEFFICIENT_PATH))
        print (u'threshold lsq coefficient loaded')
    else:
        print (u'threshold lsq coefficient not exist in %s' % THRESHOLD_LSQ_COEFFICIENT_PATH)


def derive_tag_indices_from_y(y, is_y_true=False, threshold=0.15):
    ans = set()
    if THRESHOLD_LSQ_COEFFICIENT is not None:
        threshold = sum(numpy.append(y, [1.0]) * THRESHOLD_LSQ_COEFFICIENT)
    # threshold = LR_MODEL.predict(numpy.reshape(y, newshape=(1, 88)))
    threshold = 0.01 if is_y_true else threshold
    for idx in range(len(y)):
        if y[idx] >= threshold:
            ans.add(idx)
    return ans


def main():
    import refined_preprocessing
    model = new_model(1)
    # model.save_weights(u'../models/model_weights.h5')
    print (model.predict(numpy.ones(shape=(1,1536))))

if __name__ == '__main__':
    main()
    pass
