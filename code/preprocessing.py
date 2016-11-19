# coding=utf8
import os
import re
import sys
import itertools

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
                tags = [seg for seg in url.split(u'/') if seg.isalpha()]
                tags = [u'-'.join(tags[:i + 1]) for i in range(len(tags))]
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


def get_nyt_dict(structured_nyt_path, output_dict_path):
    from codecs import open
    nyt_dict = {u'': sys.maxint}
    for line in open(structured_nyt_path, encoding='utf8'):
        for word in line.split():
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


def get_nyt_word_embeddings(nyt_dict_path, output_embedding_path):
    from gensim.models import word2vec
    model = word2vec.Word2Vec.load_word2vec_format(ur'..\models\GoogleNews-vectors-negative300.bin', binary=True)
    pass


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


if __name__ == '__main__':
    # merge_nyt_to_single_file(NYT_PATH, NYT_SINGLE_FILE_PATH)
    # structure_nyt_news_from_single_file(NYT_SINGLE_FILE_PATH,
    #                                     STRUCTURED_NYT_PATH, STRUCTURED_NYT_STAT_PATH)
    # calc_words_less_than_max_percentage()
    # calc_sentences_less_than_max_percentage()
    # get_nyt_dict(STRUCTURED_NYT_PATH, NYT_DICT_PATH)
    pass
