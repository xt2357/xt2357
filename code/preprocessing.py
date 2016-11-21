# coding=utf8
import os
import re
import sys
import itertools
import codecs
import numpy
from keras.utils import np_utils
import nlp_utils

NYT_PATH = ur'D:\nyt\NYT'
NYT_SINGLE_FILE_PATH = ur'../data/nyt/nyt_single.txt'
STRUCTURED_NYT_PATH = ur'../data/nyt/structured_nyt.txt'
STRUCTURED_NYT_STAT_PATH = ur'../data/nyt/statistic.txt'


# noinspection PyBroadException
def merge_nyt_to_single_file(nyt_path, output_path):
    from codecs import open
    count = 0
    fail_count = 0
    line_pattern = re.compile(ur'.*[a-zA-Z0-9]+.*', re.UNICODE)
    with open(output_path, 'w', encoding='utf8') as out:
        for root, dirs, files in os.walk(nyt_path):
            for filename in files:
                if filename.endswith(u'.info'):
                    info_path = os.path.join(root, filename)
                    content_path = os.path.join(root, filename[:-4] + u'txt')
                    try:
                        with open(info_path, encoding='utf8') as info, open(content_path, encoding='utf8') as content:
                            title = info.readline().split(':')[-1]
                            assert line_pattern.match(title.strip()), u'%s, title: %s' % (info_path, title)
                            url = info.readline()[5:] + u'\n'
                            assert url.startswith(u'http'), u'%s, url: %s' % (info_path, url)
                            paragraphs = [line.strip() for line in content.readlines()
                                          if line_pattern.match(line.strip())]
                            out.write(title.strip() + u'\n')
                            out.write(url.strip() + u'\n')
                            out.write(u'\n'.join(paragraphs))
                            out.write(u'\n\n')
                        count += 1
                        if (count % 10000) == 0:
                            print (u"%dw news merged.." % (count / 10000))
                    except BaseException, e:
                        print (unicode(e))
                        fail_count += 1
                        print (u'file fail: %s' % content_path)
                        if (fail_count % 100) == 0:
                            print (u"%d news failed.." % fail_count)
    print ("fail %d" % fail_count)


def extract_tags_from_url(url):
    tags = [seg for seg in url.split(u'/') if seg.replace(u'-', u'').isalpha()]
    tags = [u'#'.join(tags[:i + 1]) for i in range(len(tags))]
    if len(tags) > 3:
        tags = tags[:3]
    assert len(tags) > 0, u'tag length == 0!'
    return tags


def structure_nyt_news_from_single_file(nyt_single_file_path, output_path, statistics_output_path):
    from codecs import open
    done_count = 0
    sen_cnt, word_aver, word_min, word_max = 0, 0.0, sys.maxint, 0
    para_cnt, sen_aver, sen_min, sen_max = 0, 0.0, sys.maxint, 0
    news_cnt, para_aver, para_min, para_max = 0, 0.0, sys.maxint, 0
    with open(nyt_single_file_path, encoding='utf8') as news_file, \
            open(output_path, 'w', encoding='utf8') as out_file, \
            open(statistics_output_path, 'w', encoding='utf8') as statistic_file:
        title, url, text = u'', u'', u''
        tags = []
        # all_lines = news_file.readlines()
        for line in news_file:
            if title == u'':
                title = line
            elif url == u'':
                url = line
                assert url.startswith(u'http'), u'title: %s, url: %s' % (title, url)
                tags = extract_tags_from_url(url)
            elif line.strip() == u'':
                assert title.strip() != u'' and url.strip() != u'' and text.strip() != u'', \
                    u'title: %s, url: %s, text: %s' % (title, url, text)
                news_structure = nlp_utils.split_into_paragraph_sentence_token(title.strip() + u'\n' + text)
                out_file.write(u' '.join(tags) + u'\n')
                statistic_file.write(title + url)
                statistic_file.write(
                    u' '.join([unicode(len(news_structure))] + [unicode(len(para)) for para in news_structure]) + u'\n')
                for para in news_structure:
                    for sentence in para:
                        out_file.write(u' '.join(sentence) + u'\n')
                        sen_cnt, word_aver, word_min, word_max = \
                            sen_cnt + 1, (word_aver * sen_cnt + len(sentence)) / (sen_cnt + 1), \
                            min(word_min, len(sentence)), max(word_max, len(sentence))
                    para_cnt, sen_aver, sen_min, sen_max = \
                        para_cnt + 1, (sen_aver * para_cnt + len(para)) / (para_cnt + 1), \
                        min(sen_min, len(para)), max(sen_max, len(para))
                news_cnt, para_aver, para_min, para_max = \
                    news_cnt + 1, (para_aver * news_cnt + len(news_structure)) / (news_cnt + 1), \
                    min(para_min, len(news_structure)), max(para_max, len(news_structure))
                title, url, text = u'', u'', u''
                tags = []
                done_count += 1
                if (done_count % 1000) == 0:
                    print (u'%dk news done..' % (done_count / 1000))
                    # break
            else:
                text += line
        statistic_file.write(u'\n')
        statistic_file.write(u'sentence cnt: %d, word average: %f, min: %d, max: %d\n'
                             u'paragraph cnt: %d, sentence average: %f, min: %d, max: %d\n'
                             u'news cnt: %d, paragraph average: %f, min: %d, max: %d\n' %
                             (sen_cnt, word_aver, word_min, word_max,
                              para_cnt, sen_aver, sen_min, sen_max,
                              news_cnt, para_aver, para_min, para_max))


NYT_DICT_PATH = ur'../data/nyt/nyt_dict.txt'
NYT_IGNORE_CASE_DICT_PATH = ur'../data/nyt/nyt_ignore_case_dict.txt'


def get_nyt_dict(structured_nyt_path, output_dict_path, ignore_case=False):
    from codecs import open
    nyt_dict = {PADDING_WORD: sys.maxint}
    for line in open(structured_nyt_path, encoding='utf8'):
        for word in line.split():
            if ignore_case:
                word = word.lower()
            if word in nyt_dict:
                nyt_dict[word] += 1
            else:
                nyt_dict[word] = 1
    res = [item for item in nyt_dict.iteritems()]
    res.sort(lambda x, y: cmp(x[1], y[1]), reverse=True)
    with open(output_dict_path, 'w', encoding='utf8') as output:
        idx = 0
        for item in res:
            output.write(u'%d %s %d\n' % (idx, item[0], item[1]))
            idx += 1
    global DICTIONARY_LOADED
    DICTIONARY_LOADED = False


NYT_TAG_DICT_PATH = ur'../data/nyt/tag_dict.txt'


def get_nyt_tag_dict(structured_nyt_stat_path, nyt_tag_dict_path):
    from codecs import open
    tag_dict = {}
    for url in itertools.islice(open(structured_nyt_stat_path, encoding='utf8'), 1, None, 3):
        if not url.startswith(u'http'):
            break
        tags = extract_tags_from_url(url)
        for tag in tags:
            if tag in tag_dict:
                tag_dict[tag] += 1
            else:
                tag_dict[tag] = 1
    sorted_tags = [item for item in tag_dict.iteritems()]
    sorted_tags.sort(lambda x, y: cmp(x[1], y[1]), reverse=True)
    idx = 0
    with open(nyt_tag_dict_path, 'w', encoding='utf8') as dict_output:
        for item in sorted_tags:
            dict_output.write(u'%d %s %d\n' % (idx, item[0], item[1]))
            idx += 1
    global DICTIONARY_LOADED
    DICTIONARY_LOADED = False


NYT_WORD_EMBEDDING_PATH = ur'../data/nyt/nyt_word_embedding.txt'
NYT_IGNORE_CASE_WORD_EMBEDDING_PATH = ur'../data/nyt/nyt_ignore_case_word_embedding.txt'
NYT_WORD_EMBEDDING_DIM = 300


def get_nyt_word_embeddings(nyt_dict_path, output_embedding_path):
    from gensim.models import word2vec
    from codecs import open
    model = word2vec.Word2Vec.load_word2vec_format(ur'..\models\GoogleNews-vectors-negative300.bin', binary=True)
    word_embedding_dim = NYT_WORD_EMBEDDING_DIM
    with open(output_embedding_path, 'w', encoding='utf8') as output:
        v = [0.0] * word_embedding_dim
        for line in open(nyt_dict_path, encoding='utf8'):
            word = line.split()[1]
            output.write(line.strip() + u' ')
            if word == PADDING_WORD or word not in model:
                output.write(u' '.join([str(component) for component in v]) + u'\n')
            else:
                output.write(u' '.join([str(component) for component in model[word]]) + u'\n')


# (21946236, 21341986) words in sentences: (sentence cnt, less than 48 cnt)
# (600929, 478460) sentences in documents (document cnt, less than 48 cnt)
MAX_WORDS_IN_SENTENCE = 48
MAX_SENTENCES_IN_DOCUMENT = 48


def calc_words_less_than_max_percentage():
    line_cnt, my_cnt = 0, 0
    for line in file(STRUCTURED_NYT_PATH):
        if len(line.split()) <= MAX_WORDS_IN_SENTENCE:
            my_cnt += 1
        line_cnt += 1
    print (line_cnt, my_cnt)


def calc_sentences_less_than_max_percentage():
    doc_cnt, ok_cnt = 0, 0
    for stat in itertools.islice(file(STRUCTURED_NYT_STAT_PATH), 2, None, 3):
        if not stat[0].isdigit():
            break
        doc_cnt += 1
        if sum([int(c) for c in stat.split()[1:]]) <= MAX_SENTENCES_IN_DOCUMENT:
            ok_cnt += 1
    print (doc_cnt, ok_cnt)


DICTIONARY_LOADED = False
DICT_SIZE = 200000
# use specific string for padding when word are not in dictionary
PADDING_WORD = u'-'
PADDING_WORD_IDX = 0
DICTIONARY = {}
TAG_DICT_SIZE = 0
TAG_DICTIONARY = {}


def load_dictionaries():
    global DICTIONARY_LOADED
    if DICTIONARY_LOADED:
        return
    global TAG_DICT_SIZE
    print (u'loading dictionary from %s..' % NYT_IGNORE_CASE_DICT_PATH)
    _cnt = 0
    for dictionary_line in codecs.open(NYT_IGNORE_CASE_DICT_PATH, encoding='utf8'):
        if _cnt >= DICT_SIZE:
            break
        _word = dictionary_line.split()[1]
        _idx = int(dictionary_line.split()[0])
        DICTIONARY[_word] = _idx
        _cnt += 1
    print (u'loading dictionary from %s done..' % NYT_IGNORE_CASE_DICT_PATH)

    print (u'loading tag dictionary from %s..' % NYT_TAG_DICT_PATH)
    for tag_dictionary_line in codecs.open(NYT_TAG_DICT_PATH, encoding='utf8'):
        _idx = int(tag_dictionary_line.split()[0])
        _tag = tag_dictionary_line.split()[1]
        TAG_DICTIONARY[_tag] = _idx
        TAG_DICT_SIZE += 1
    print (u'loading tag dictionary from %s done..' % NYT_TAG_DICT_PATH)
    DICTIONARY_LOADED = True


# load dictionary when import
load_dictionaries()


# sentences: [[word1, word2, ...], [], [], ..]
def padding_document(sentences):
    if len(sentences) > MAX_SENTENCES_IN_DOCUMENT:
        sentences = sentences[:MAX_SENTENCES_IN_DOCUMENT]
    elif len(sentences) < MAX_SENTENCES_IN_DOCUMENT:
        nb_sentences_padding = MAX_SENTENCES_IN_DOCUMENT - len(sentences)
        sentences = [[PADDING_WORD] * MAX_WORDS_IN_SENTENCE] * nb_sentences_padding + sentences
    for i in range(len(sentences)):
        nb_words = len(sentences[i])
        if nb_words > MAX_WORDS_IN_SENTENCE:
            sentences[i] = sentences[i][:MAX_WORDS_IN_SENTENCE]
        elif nb_words < MAX_WORDS_IN_SENTENCE:
            nb_words_padding = MAX_WORDS_IN_SENTENCE - nb_words
            sentences[i] = [PADDING_WORD] * nb_words_padding + sentences[i]
        sentences[i] = [DICTIONARY[word] if word in DICTIONARY else PADDING_WORD_IDX for word in sentences[i]]
    return reduce(list.__add__, sentences)


X_ALL_PATH = ur'../data/nyt/x_all.txt'
Y_ALL_PATH = ur'../data/nyt/y_all.txt'


def transform_structured_nyt_to_regular_data(structured_nyt_path, structured_nyt_stat_path, x_all_path, y_all_path):
    load_dictionaries()
    from codecs import open
    with open(structured_nyt_stat_path, encoding='utf8') as stat, \
            open(structured_nyt_path, encoding='utf8') as nyt, \
            open(x_all_path, 'w', encoding='utf8') as x_all, \
            open(y_all_path, 'w', encoding='utf8') as y_all:
        for stat_line in itertools.islice(stat, 2, None, 3):
            if not stat_line[0].isdigit():
                break
            nb_sentence = sum([int(cnt) for cnt in stat_line.split()[1:]])
            y_all.write(u' '.join([str(TAG_DICTIONARY[tag]) for tag in nyt.readline().split()]) + u'\n')
            sentences = []
            for i in range(nb_sentence):
                sentences.append(nyt.readline().split())
            x_all.write(u' '.join([str(idx) for idx in padding_document(sentences)]) + u'\n')


X_TRAIN_PATH = ur'../data/nyt/x_train.txt'
Y_TRAIN_PATH = ur'../data/nyt/y_train.txt'
X_EVAL_PATH = ur'../data/nyt/x_eval.txt'
Y_EVAL_PATH = ur'../data/nyt/y_eval.txt'


def randomly_split_data(eval_data_size, x_all_path, y_all_path, x_train_path, y_train_path, x_eval_path, y_eval_path):
    from random import randint
    reservoir = []
    with codecs.open(x_all_path, encoding='utf8') as x_all, \
            codecs.open(y_all_path, encoding='utf8') as y_all, \
            codecs.open(x_train_path, 'w', encoding='utf8') as x_train, \
            codecs.open(y_train_path, 'w', encoding='utf8') as y_train:
        idx = 0
        for x in x_all:
            y = y_all.readline()
            if idx < eval_data_size:
                reservoir.append((x, y))
            else:
                r = randint(0, idx)
                if r < eval_data_size:
                    x_train.write(reservoir[r][0])
                    y_train.write(reservoir[r][1])
                    reservoir[r] = (x, y)
                else:
                    x_train.write(x)
                    y_train.write(y)
            idx += 1
            if (idx % 10000) == 0:
                print (u'%dw samples done..' % (idx / 10000))
    with codecs.open(x_eval_path, 'w', encoding='utf8') as x_eval, \
            codecs.open(y_eval_path, 'w', encoding='utf8') as y_eval:
        for (x, y) in reservoir:
            x_eval.write(x)
            y_eval.write(y)


def read_x(file_path):
    cnt = 0
    for _ in open(file_path):
        cnt += 1
    ans = numpy.zeros((cnt, MAX_WORDS_IN_SENTENCE * MAX_SENTENCES_IN_DOCUMENT))
    i = 0
    for line in open(file_path):
        ans[i] = numpy.asarray(line.split(), dtype='int32')
        i += 1
    return ans


def read_y(file_path):
    cnt = 0
    for _ in open(file_path):
        cnt += 1
    ans = numpy.zeros((cnt, TAG_DICT_SIZE))
    i = 0
    for line in open(file_path):
        # normalization of the distribution
        points = [int(idx) for idx in line.split()]
        v = numpy.zeros((TAG_DICT_SIZE,), dtype='float32')
        for idx in points:
            v[idx] = 1.0 / len(points)
        ans[i] = v
        i += 1
    return ans


if __name__ == '__main__':
    # merge_nyt_to_single_file(NYT_PATH, NYT_SINGLE_FILE_PATH)
    # structure_nyt_news_from_single_file(NYT_SINGLE_FILE_PATH,
    #                                     STRUCTURED_NYT_PATH, STRUCTURED_NYT_STAT_PATH)
    # calc_words_less_than_max_percentage()
    # calc_sentences_less_than_max_percentage()

    # get_nyt_dict(STRUCTURED_NYT_PATH, NYT_DICT_PATH)
    # get_nyt_word_embeddings(NYT_DICT_PATH, NYT_WORD_EMBEDDING_PATH)

    # get_nyt_dict(STRUCTURED_NYT_PATH, NYT_IGNORE_CASE_DICT_PATH, ignore_case=True)
    # get_nyt_word_embeddings(NYT_IGNORE_CASE_DICT_PATH, NYT_IGNORE_CASE_WORD_EMBEDDING_PATH)

    # print (padding_document([[u'hello', u'world'], [], [u'a', u'b']]))
    # get_nyt_tag_dict(STRUCTURED_NYT_STAT_PATH, NYT_TAG_DICT_PATH)

    # transform_structured_nyt_to_regular_data(STRUCTURED_NYT_PATH, STRUCTURED_NYT_STAT_PATH, X_ALL_PATH, Y_ALL_PATH)
    # randomly_split_data(100000, X_ALL_PATH, Y_ALL_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, X_EVAL_PATH, Y_EVAL_PATH)
    pass
