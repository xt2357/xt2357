# coding=utf8
from gensim.models import word2vec



PATH = ur'D:\nyt\NYT\arts\2009\01\23\Inside Art - Clare and Eugene V. Thaw Promise Oil Sketches to Metropolitan Museum and Morgan Library; Gifts to Philadelphia Museum of Art Honor Anne d’Harnoncourt; Nahmad Family Is Buyer of Kazimir Malevich Painting; New Category for Christie’s.info'
REAL_PATH = ur'D:\nyt\NYT\arts\2009\01\23\Inside Art - Clare and Eugene V. Thaw Promise Oil Sketches to Metropolitan Museum and Morgan Library; Gifts to Philadelphia Museum of Art Honor Anne d’Harnoncourt; Nahmad Family Is Buyer of Kazimir Malevich Painting; New Category for Christie’s.info'


def test():
    # from codecs import open
    f = open(REAL_PATH)
    print (f.readlines())


def test_gensim():
    model = word2vec.Word2Vec.load_word2vec_format(ur'..\models\GoogleNews-vectors-negative300.bin', binary=True)
    print (model[ur'sometimes'])
    print (model[ur'Sometimes'])
    while True:
        word = raw_input()
        if word in model:
            print (model[word])


if __name__ == '__main__':
    test_gensim()
