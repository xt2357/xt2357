# coding=utf8
import lstm_with_tag_relation as my_model
import preprocessing


def main():
    model = my_model.masked_simplified_lstm(preprocessing.MAX_WORDS_IN_SENTENCE, preprocessing.MAX_WORDS_IN_SENTENCE,
                                            20000, None, 300, 600, 100)
    model.get_layer(name=u'relation')


if __name__ == '__main__':
    main()
