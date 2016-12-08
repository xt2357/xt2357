# coding=utf8
"""
filter meaningful tags
"""
from preprocessing import *
import numpy

REFINED_NYT_TAG_DICT_PATH = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_tag_dict.txt')


def get_refined_nyt_tag_dict(structured_nyt_stat_path, refined_nyt_tag_dict_path):
    from codecs import open
    tag_dict = {}
    for url in itertools.islice(open(structured_nyt_stat_path, encoding='utf8'), 1, None, 3):
        if not url.startswith(u'http'):
            break
        tags = extract_tags_from_url(url)
        # remove too long path tags && fix path length to 2
        if len(tags) >= 3:
            continue
        elif len(tags) == 1:
            tags.append(tags[0] + u'#notanyoneofsubtags')
        for tag in tags:
            if tag in tag_dict:
                tag_dict[tag] += 1
            else:
                tag_dict[tag] = 1
    sorted_tags = [item for item in tag_dict.iteritems()]
    sorted_tags.sort(lambda x, y: cmp(x[1], y[1]), reverse=True)
    sorted_tags = [item for item in sorted_tags if item[1] >= 1000]
    tag_set = set([item[0] for item in sorted_tags])
    subtag_of = {}
    for tag in tag_set:
        if tag.find(u'#') == -1:
            continue
        first = tag.split(u'#')[0]
        if first not in subtag_of:
            subtag_of[first] = set()
        subtag_of[first].add(tag)
        subtag_of[tag] = set()
    for k, v in list(subtag_of.iteritems()):
        if k in subtag_of and len(subtag_of[k]) == 1:
            for ele in subtag_of[k]:
                del subtag_of[ele]
            subtag_of[k] = set()
    print subtag_of
    idx = 0
    with open(refined_nyt_tag_dict_path, 'w', encoding='utf8') as dict_output:
        sorted_tags.sort(lambda x, y: 1 if x[0].find(u'#') != -1 else -1)
        for item in sorted_tags:
            if item[0] not in subtag_of:
                continue
            dict_output.write(u'%d %s %d\n' % (idx, item[0], item[1]))
            idx += 1


class TagManager(object):
    REFINED_TAG_DICT_SIZE = 0
    REFINED_TAG_TO_IDX = {}
    IDX_TO_REFINED_TAG = {}
    BIG_TAG_TO_SEQ = {}
    SUB_TAG_TO_SEQ = {}
    SEQ_TO_BIG_TAG = {}
    SEQ_TO_SUB_TAG = {}
    MAX_SUB_TAG_COUNT = 999
    SUBTAG_COUNT = []
    BIG_TAG_COUNT = 0

    @classmethod
    def init(cls):
        for line in open(REFINED_NYT_TAG_DICT_PATH):
            tag_name = line.split()[1]
            if tag_name.find(u'#') == -1:
                cls.BIG_TAG_COUNT += 1
        big_seq = 0
        cls.SUBTAG_COUNT = [0] * cls.BIG_TAG_COUNT
        idx = 0
        for line in open(REFINED_NYT_TAG_DICT_PATH):
            cls.REFINED_TAG_DICT_SIZE += 1
            tag_name = line.split()[1]
            cls.IDX_TO_REFINED_TAG[idx] = tag_name
            cls.REFINED_TAG_TO_IDX[tag_name] = idx
            idx += 1
            if tag_name.find(u'#') == -1:
                cls.BIG_TAG_TO_SEQ[tag_name] = big_seq
                cls.SEQ_TO_BIG_TAG[big_seq] = tag_name
                big_seq += 1
            else:
                big_tag = tag_name.split(u'#')[0]
                big_tag_seq = cls.BIG_TAG_TO_SEQ[big_tag]
                cls.SUB_TAG_TO_SEQ[tag_name] = cls.SUBTAG_COUNT[big_tag_seq]
                if big_tag not in cls.SEQ_TO_SUB_TAG:
                    cls.SEQ_TO_SUB_TAG[big_tag] = {}
                cls.SEQ_TO_SUB_TAG[big_tag][cls.SUBTAG_COUNT[big_tag_seq]] = tag_name
                cls.SUBTAG_COUNT[big_tag_seq] += 1
        print (u'refined tag manager loaded..')

    @classmethod
    def has_tag(cls, tag_name):
        return tag_name in cls.REFINED_TAG_TO_IDX

    @classmethod
    def is_sub_tag(cls, encoding):
        return (encoding % (cls.MAX_SUB_TAG_COUNT + 1)) != cls.MAX_SUB_TAG_COUNT

    @classmethod
    def idx(cls, tag_name):
        return cls.REFINED_TAG_TO_IDX[tag_name]

    @classmethod
    def encode(cls, tag_name):
        big_tag = tag_name.split(u'#')[0]
        sub_tag = None if tag_name.find(u'#') == -1 else tag_name
        if sub_tag:
            return cls.BIG_TAG_TO_SEQ[big_tag] * (cls.MAX_SUB_TAG_COUNT + 1) + cls.SUB_TAG_TO_SEQ[sub_tag]
        else:
            return cls.BIG_TAG_TO_SEQ[big_tag] * (cls.MAX_SUB_TAG_COUNT + 1) + cls.MAX_SUB_TAG_COUNT


TagManager.init()
REFINED_X_ALL_PATH = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_x_all.txt')
REFINED_Y_ALL_PATH = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_y_all.txt')


def transform_structured_nyt_to_refined_regular_data(structured_nyt_path, structured_nyt_stat_path,
                                                     x_all_path, y_all_path):
    load_dictionaries()
    from codecs import open
    with open(structured_nyt_stat_path, encoding='utf8') as stat, \
            open(structured_nyt_path, encoding='utf8') as nyt, \
            open(x_all_path, 'w', encoding='utf8') as x_all, \
            open(y_all_path, 'w', encoding='utf8') as y_all:
        finished = 0
        for stat_line in itertools.islice(stat, 2, None, 3):
            if not stat_line[0].isdigit():
                break
            nb_sentence = sum([int(cnt) for cnt in stat_line.split()[1:]])
            tags = [tag for tag in nyt.readline().split() if TagManager.has_tag(tag)]
            if len(tags) == 1 and TagManager.SUBTAG_COUNT[TagManager.BIG_TAG_TO_SEQ[tags[0]]] != 0:
                tags.append(tags[0] + u'#notanyoneofsubtags')
            tags = [(TagManager.encode(tag), TagManager.idx(tag))
                    for tag in tags if TagManager.has_tag(tag)]
            sentences = []
            for i in range(nb_sentence):
                sentences.append(nyt.readline().split())
            if len(tags) == 0:
                continue
            x_all.write(u' '.join([str(idx) for idx in padding_document(sentences)]) + u'\n')
            y_all.write(u' '.join([u'%s,%s' % (tag[0], tag[1]) for tag in tags]) + u'\n')
            finished += 1
            if finished % 10000 == 0:
                print (u'%dw done..' % (finished / 10000))


def calc_average_tag_per_refined_doc():
    aver, cnt = 0.0, 0
    for y in open(REFINED_Y_ALL_PATH):
        cnt += 1
        aver = (aver * cnt - aver + len(y.split())) / cnt
    print (aver, cnt)


def randomly_split_data_in_memory(fraction, x_all, y_all):
    from random import randint
    eval_data_size = int(len(x_all) * fraction)
    res = {u'x_train': numpy.zeros(shape=(len(x_all) - eval_data_size,
                                          len(x_all[0]) if hasattr(x_all[0], '__len__') else 1)),
           u'y_train': numpy.zeros(shape=(len(y_all) - eval_data_size,
                                          len(y_all[0]) if hasattr(y_all[0], '__len__') else 1)),
           u'x_eval': numpy.zeros(shape=(eval_data_size, len(x_all[0]) if hasattr(x_all[0], '__len__') else 1)),
           u'y_eval': numpy.zeros(shape=(eval_data_size, len(y_all[0]) if hasattr(y_all[0], '__len__') else 1))}
    reservoir = []
    idx = 0
    train_idx = 0
    for x in x_all:
        y = y_all[idx]
        if idx < eval_data_size:
            reservoir.append((x, y))
        else:
            r = randint(0, idx)
            if r < eval_data_size:
                res[u'x_train'][train_idx] = reservoir[r][0]
                res[u'y_train'][train_idx] = reservoir[r][1]
                reservoir[r] = (x, y)
            else:
                res[u'x_train'][train_idx] = x
                res[u'y_train'][train_idx] = y
            train_idx += 1
        idx += 1
        if (idx % 10000) == 0:
            print (u'split samples in memory: %d0w samples done..' % (idx / 100000))
    eval_idx = 0
    for sample in reservoir:
        res[u'x_eval'][eval_idx] = sample[0]
        res[u'y_eval'][eval_idx] = sample[1]
    return res


def read_refined_x(file_path, size=None):
    cnt = 0
    for _ in open(file_path):
        if size and cnt == size:
            break
        cnt += 1
    ans = numpy.zeros((cnt, MAX_WORDS_IN_SENTENCE * MAX_SENTENCES_IN_DOCUMENT))
    i = 0
    for line in open(file_path):
        ans[i] = numpy.asarray(line.split(), dtype='int32')
        i += 1
        if i == cnt:
            break
    return ans


def read_refined_y(file_path, size=None, return_idx=False):
    cnt = 0
    for _ in open(file_path):
        if size and cnt == size:
            break
        cnt += 1
    # at most 2 codings
    dim = TagManager.REFINED_TAG_DICT_SIZE if return_idx else 2
    ans = numpy.zeros((cnt, dim))
    i = 0
    for line in open(file_path):
        # read as a multi-label y
        if return_idx:
            # normalization of the distribution
            points = [int(pair.split(u',')[1]) for pair in line.split()]
            v = numpy.zeros((TagManager.REFINED_TAG_DICT_SIZE,), dtype='float32')
            for idx in points:
                v[idx] = 1.0 / len(points)
            ans[i] = v
        else:
            codings = [int(pair.split(u',')[0]) for pair in line.split()]
            if len(codings) < 2:
                codings.append(-1)
            v = numpy.zeros((2,), dtype='int32')
            v[0] = codings[0]
            v[1] = codings[1]
            ans[i] = v
        i += 1
        if i == cnt:
            break
    return ans


# this function will be used only when applying hierarchical classification
def filter_x_y(x, y, related_big_tag):
    if related_big_tag == u'all_big_tags':
        res_y = numpy.zeros((len(y), TagManager.BIG_TAG_COUNT))
        idx = 0
        for row in y:
            encode = row[0]
            v = numpy.zeros((TagManager.BIG_TAG_COUNT,), dtype='float32')
            v[int(encode / (TagManager.MAX_SUB_TAG_COUNT + 1))] = 1.0
            res_y[idx] = v
            idx += 1
        return x, res_y
    else:
        big_tag_seq = TagManager.BIG_TAG_TO_SEQ[related_big_tag]
        dim = TagManager.SUBTAG_COUNT[big_tag_seq]
        res_x = numpy.zeros((len(x), len(x[0])))
        res_y = numpy.zeros((len(y), dim))
        idx = 0
        res_idx = 0
        for row in y:
            # first encode stands for the big tag, the second one is sub tag or -1 if there is no sub tag
            encode = row[1]
            if encode == -1 or int(encode / (TagManager.MAX_SUB_TAG_COUNT + 1)) != big_tag_seq:
                idx += 1
                continue
            v = numpy.zeros((dim,), dtype='float32')
            v[int(encode % (TagManager.MAX_SUB_TAG_COUNT + 1))] = 1.0
            res_y[res_idx] = v
            res_x[res_idx] = x[idx]
            res_idx += 1
            idx += 1
        print (u'filter %d samples of big tag: %s from %d samples' % (res_idx, related_big_tag, len(x)))
        return res_x[:res_idx], res_y[:res_idx]


REFINED_X_TRAIN = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_x_train.txt')
REFINED_Y_TRAIN = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_y_train.txt')
REFINED_X_EVAL = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_x_eval.txt')
REFINED_Y_EVAL = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_y_eval.txt')


# special splitted data: for every big tag, randomly split 10% of its data into eval data set
REFINED_X_TRAIN_SP = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_x_train_sp.txt')
REFINED_Y_TRAIN_SP = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_y_train_sp.txt')
REFINED_X_EVAL_SP = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_x_eval_sp.txt')
REFINED_Y_EVAL_SP = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/refined_y_eval_sp.txt')


def sp_split_all_refined_data():
    def reservoir_sampling(x_all, y_all, size):
        from random import randint
        reservoir = []
        x_train, x_eval, y_train, y_eval = [], [], [], []
        idx = 0
        for x in x_all:
            y = y_all[idx]
            if idx < size:
                reservoir.append((x, y))
            else:
                r = randint(0, idx)
                if r < size:
                    x_train.append(reservoir[r][0])
                    y_train.append(reservoir[r][1])
                    reservoir[r] = (x, y)
                else:
                    x_train.append(x)
                    y_train.append(y)
            idx += 1
        for x, y in reservoir:
            x_eval.append(x)
            y_eval.append(y)
        return {u'x_train': x_train, u'x_eval': x_eval, u'y_train': y_train, u'y_eval': y_eval}

    def filter_by_big_tag(x_all, y_all, big_tag_seq):
        res_x, res_y = [], []
        idx = 0
        for x in x_all:
            y = y_all[idx]
            if int(y.split()[0].split(u',')[0]) / (TagManager.MAX_SUB_TAG_COUNT + 1) == big_tag_seq:
                res_x.append(x)
                res_y.append(y)
            idx += 1
        return res_x, res_y

    refined_x_all = open(REFINED_X_ALL_PATH).readlines()
    refined_y_all = open(REFINED_Y_ALL_PATH).readlines()
    with open(REFINED_X_TRAIN_SP, 'w') as x_train_sp, \
            open(REFINED_Y_TRAIN_SP, 'w') as y_train_sp, \
            open(REFINED_X_EVAL_SP, 'w') as x_eval_sp, \
            open(REFINED_Y_EVAL_SP, 'w') as y_eval_sp:
        for big_tag, seq in TagManager.BIG_TAG_TO_SEQ.items():
            print (u'%s processing' % big_tag)
            filtered_x, filtered_y = filter_by_big_tag(refined_x_all, refined_y_all, seq)
            datum = reservoir_sampling(filtered_x, filtered_y, int(len(filtered_x) * 0.1))
            for line in datum[u'x_train']:
                x_train_sp.write(line)
            for line in datum[u'y_train']:
                y_train_sp.write(line)
            for line in datum[u'x_eval']:
                x_eval_sp.write(line)
            for line in datum[u'y_eval']:
                y_eval_sp.write(line)


if __name__ == '__main__':
    # get_refined_nyt_tag_dict(STRUCTURED_NYT_STAT_IGNORE_STOP_PATH, REFINED_NYT_TAG_DICT_PATH)
    transform_structured_nyt_to_refined_regular_data(STRUCTURED_NYT_PATH, STRUCTURED_NYT_STAT_PATH,
                                                     REFINED_X_ALL_PATH, REFINED_Y_ALL_PATH)

    import preprocessing
    preprocessing.randomly_split_data(50000, REFINED_X_ALL_PATH, REFINED_Y_ALL_PATH,
                                      REFINED_X_TRAIN, REFINED_Y_TRAIN, REFINED_X_EVAL, REFINED_Y_EVAL)
    calc_average_tag_per_refined_doc()
    sp_split_all_refined_data()
    pass
