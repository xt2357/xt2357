# coding=utf8
import lstm_with_tag_relation as my_model
import preprocessing


def main():
    model = my_model.masked_simplified_lstm(preprocessing.MAX_SENTENCES_IN_DOCUMENT,
                                            preprocessing.MAX_WORDS_IN_SENTENCE,
                                            preprocessing.DICT_SIZE, None, 300, 600, preprocessing.TAG_DICT_SIZE)
    model.get_layer(name=u'relation')


if __name__ == '__main__':
    main()
