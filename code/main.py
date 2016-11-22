# coding=utf8
import sys
import lstm_with_tag_relation as my_model
import preprocessing
import nlp_utils
import codecs
import numpy
from keras.callbacks import EarlyStopping


MODEL_WEIGHTS_PATH = u'../models/model_weights.h5'
HISTORY_PATH = u'../models/train_history.txt'


def train():
    model = my_model.masked_simplified_lstm(preprocessing.MAX_SENTENCES_IN_DOCUMENT,
                                            preprocessing.MAX_WORDS_IN_SENTENCE,
                                            preprocessing.DICT_SIZE,
                                            my_model.get_embedding_weights(
                                                preprocessing.NYT_IGNORE_CASE_WORD_EMBEDDING_PATH),
                                            preprocessing.NYT_WORD_EMBEDDING_DIM, 400, preprocessing.TAG_DICT_SIZE)
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


def text_predict(trained_model, text, threshold):
    # replace all the \n to make the whole text a single paragraph
    structure = nlp_utils.split_into_paragraph_sentence_token(text.replace(u'\n', u''))
    sentences = structure[0]
    input_v = preprocessing.padding_document(sentences)
    output_v = trained_model.predict([input_v])[0]
    output_v *= output_v > threshold
    idx = 0
    tags = []
    for confidence in output_v:
        if confidence > 0.0:
            tags.append((idx, confidence))
        idx += 1
    return [(preprocessing.TAG_IDX_TO_NAME[idx], confidence) for idx, confidence in tags]


def sample_based_validation(x_eval, y_eval, trained_model, threshold):
    y_pred = trained_model.predict(x_eval)
    y_pred *= y_pred > threshold
    numpy.ceil(y_pred, y_pred)
    numpy.ceil(y_eval, y_eval)
    symmetric_diff = numpy.abs(y_pred - y_eval)
    print (numpy.sum(numpy.sum(symmetric_diff, axis=-1).clip(0.0, 1.0)) / x_eval.shape[0])
    print (u'-------hamming loss---------')
    print (numpy.sum(symmetric_diff) / x_eval.shape[0])


THRESHOLD = 1 / 5.0


def evaluation():
    model = my_model.masked_simplified_lstm(preprocessing.MAX_SENTENCES_IN_DOCUMENT,
                                            preprocessing.MAX_WORDS_IN_SENTENCE,
                                            preprocessing.DICT_SIZE,
                                            my_model.get_embedding_weights(
                                                preprocessing.NYT_IGNORE_CASE_WORD_EMBEDDING_PATH),
                                            preprocessing.NYT_WORD_EMBEDDING_DIM, 400, preprocessing.TAG_DICT_SIZE)
    model.load_weights(MODEL_WEIGHTS_PATH)
    x_eval, y_eval = \
        preprocessing.read_x(preprocessing.X_EVAL_PATH), preprocessing.read_y(preprocessing.Y_EVAL_PATH)
    print (u'eval data loaded')
    sample_based_validation(x_eval, y_eval, model, THRESHOLD)


def print_relation():
    model = my_model.masked_simplified_lstm(preprocessing.MAX_SENTENCES_IN_DOCUMENT,
                                            preprocessing.MAX_WORDS_IN_SENTENCE,
                                            preprocessing.DICT_SIZE,
                                            my_model.get_embedding_weights(
                                                preprocessing.NYT_IGNORE_CASE_WORD_EMBEDDING_PATH),
                                            preprocessing.NYT_WORD_EMBEDDING_DIM, 400, preprocessing.TAG_DICT_SIZE)
    model.load_weights(MODEL_WEIGHTS_PATH)
    print (model.get_layer(u'relation').get_weights())


def print_usage():
    print (u'usage: python main.py train for training\n'
           u'       python main.py eval for evaluation')

if __name__ == '__main__':
    print_relation()
    if len(sys.argv) < 2:
        print_usage()
    elif sys.argv[1] == u'train':
        train()
    elif sys.argv[1] == u'eval':
        evaluation()
    else:
        print_usage()

