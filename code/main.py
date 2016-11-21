# coding=utf8
import lstm_with_tag_relation as my_model
import preprocessing
import codecs
from keras.callbacks import EarlyStopping


MODEL_WEIGHTS_PATH = u'../models/model_weights.h5'
HISTORY_PATH = u'../models/train_history.txt'


def main():
    model = my_model.masked_simplified_lstm(preprocessing.MAX_SENTENCES_IN_DOCUMENT,
                                            preprocessing.MAX_WORDS_IN_SENTENCE,
                                            preprocessing.DICT_SIZE,
                                            my_model.get_embedding_weights(
                                                preprocessing.NYT_IGNORE_CASE_WORD_EMBEDDING_PATH),
                                            preprocessing.NYT_WORD_EMBEDDING_DIM, 600, preprocessing.TAG_DICT_SIZE)
    x_train, y_train = \
        preprocessing.read_x(preprocessing.X_TRAIN_PATH), preprocessing.read_y(preprocessing.Y_TRAIN_PATH)
    print (u'train data loaded')
    x_eval, y_eval = \
        preprocessing.read_x(preprocessing.X_EVAL_PATH), preprocessing.read_y(preprocessing.Y_EVAL_PATH)
    print (u'eval data loaded')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(x_train, y_train, callbacks=[early_stopping], validation_data=(x_eval, y_eval),
                        nb_epoch=2, batch_size=128)
    print (history.history)
    with codecs.open(HISTORY_PATH, 'w', 'utf8') as history_output:
        history_output.write(history.history)
    model.save_weights(MODEL_WEIGHTS_PATH)
    print (u'model saved to %s' % MODEL_WEIGHTS_PATH)


if __name__ == '__main__':
    main()
