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
import refined_preprocessing
import nlp_utils
import os
import sys
import copy
import math
import heapq
from keras.callbacks import EarlyStopping, ModelCheckpoint


def lstm_doc_embedding(nb_sentence, nb_words, dict_size, word_embedding_weights,
                       word_embedding_dim, sentence_embedding_dim, document_embedding_dim):
    word_lstm_model = Sequential()
    word_lstm_model.add(Masking(input_shape=(nb_words, word_embedding_dim), name=u'word_lstm_masking'))
    word_lstm = LSTM(output_dim=sentence_embedding_dim, input_shape=(None, word_embedding_dim),
                     activation=u'tanh', inner_activation=u'hard_sigmoid', name=u'word_lstm')
    word_lstm_model.add(word_lstm)
    sentence_lstm_model = Sequential()
    sentence_lstm_model.add(Masking(input_shape=(nb_sentence, sentence_embedding_dim), name=u'sentence_lstm_masking'))
    sentence_lstm = LSTM(output_dim=document_embedding_dim, input_shape=(None, sentence_embedding_dim),
                         activation=u'tanh', inner_activation=u'hard_sigmoid', name=u'sentence_lstm')
    sentence_lstm_model.add(sentence_lstm)

    total_words = nb_words * nb_sentence
    input_layer = Input(shape=(total_words,))
    embedding_layer = \
        Embedding(dict_size, word_embedding_dim, weights=word_embedding_weights,
                  trainable=False, name=u'word_embedding')(input_layer)
    first_reshape = Reshape((nb_sentence, nb_words, word_embedding_dim))(embedding_layer)
    sentence_embeddings = TimeDistributed(word_lstm_model)(first_reshape)
    document_embedding = sentence_lstm_model(sentence_embeddings)
    model = Model(input=input_layer, output=document_embedding)
    return model


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


def get_lstm_doc_embedding():
    return lstm_doc_embedding(preprocessing.MAX_SENTENCES_IN_DOCUMENT,
                              preprocessing.MAX_WORDS_IN_SENTENCE,
                              preprocessing.DICT_SIZE,
                              read_embedding_weights(
                                  preprocessing.NYT_IGNORE_CASE_WORD_EMBEDDING_PATH),
                              preprocessing.NYT_WORD_EMBEDDING_DIM, 450, 800)


def get_model_by_big_tag(related_big_tag):
    if related_big_tag == u'all_big_tags':
        model = Sequential()
        model.add(get_lstm_doc_embedding())
        model.add(Dense(output_dim=refined_preprocessing.TagManager.BIG_TAG_COUNT,
                        activation=u'softmax', name=related_big_tag))
        model.compile(loss=u'categorical_crossentropy', optimizer=u'rmsprop', metrics=[u'accuracy'])
        return model
    else:
        model = Sequential()
        model.add(get_lstm_doc_embedding())
        related_big_tag_seq = refined_preprocessing.TagManager.BIG_TAG_TO_SEQ[related_big_tag]
        model.add(Dense(output_dim=refined_preprocessing.TagManager.SUBTAG_COUNT[related_big_tag_seq],
                        activation=u'softmax', name=related_big_tag))
        model.compile(loss=u'categorical_crossentropy', optimizer=u'rmsprop', metrics=[u'accuracy'])
        return model


REFINED_MODEL_WEIGHTS_PATH_ROOT = os.path.join(os.path.dirname(__file__), ur'../models/refined/')
REFINED_DATA_PATH_ROOT = os.path.join(os.path.dirname(__file__), ur'../data/nyt/refined/')


def get_big_tag_model_save_path(big_tag):
    return os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'%s.h5' % big_tag)


def train_big_tag_model(big_tag, x_train, y_train, x_eval, y_eval):
    model_weights_save_path = get_big_tag_model_save_path(big_tag)
    check_point_save_path = os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'%s {epoch:02d}.h5' % big_tag)
    model = get_model_by_big_tag(big_tag)
    # load pre-trained weights
    if big_tag != u'all_big_tags':
        print (u'load all_big_tags model')
        model.load_weights(os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'all_big_tags.h5'), by_name=True)
        print (u'load all_big_tags model done..')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    check_point = ModelCheckpoint(check_point_save_path, save_weights_only=True)
    print (u'start train model for big tag: %s' % big_tag)
    history = model.fit(x_train, y_train, validation_data=(x_eval, y_eval),
                        batch_size=64, nb_epoch=4, callbacks=[early_stopping, check_point])
    model.save_weights(model_weights_save_path)
    print (u'model for big tag: %s train done, weights saved to %s' % (big_tag, model_weights_save_path))
    history_path = os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'%s history.txt' % big_tag)
    with codecs.open(history_path, 'w', 'utf8') as history_output:
        history_output.write(unicode(history.history))


def train(from_scratch=False):
    x_train, x_eval = \
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_TRAIN_SP), \
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_EVAL_SP)
    y_train, y_eval = \
        refined_preprocessing.read_refined_y(refined_preprocessing.REFINED_Y_TRAIN_SP), \
        refined_preprocessing.read_refined_y(refined_preprocessing.REFINED_Y_EVAL_SP)
    print (u'all refined x && y loaded..')
    if from_scratch:
        print (u'filtering data set for all big tags')
        all_big_tag_x_train, all_big_tag_y_train = refined_preprocessing.filter_x_y(x_train, y_train, u'all_big_tags')
        all_big_tag_x_eval, all_big_tag_y_eval = refined_preprocessing.filter_x_y(x_eval, y_eval, u'all_big_tags')
        train_big_tag_model(u'all_big_tags', all_big_tag_x_train, all_big_tag_y_train,
                            all_big_tag_x_eval, all_big_tag_y_eval)
    for big_tag, big_tag_seq in refined_preprocessing.TagManager.BIG_TAG_TO_SEQ.items():
        if refined_preprocessing.TagManager.SUBTAG_COUNT[big_tag_seq] == 0:
            continue
        print (u'filtering data set for %s' % big_tag)
        cur_x_train, cur_y_train = refined_preprocessing.filter_x_y(x_train, y_train, big_tag)
        cur_x_eval, cur_y_eval = refined_preprocessing.filter_x_y(x_eval, y_eval, big_tag)
        print (u'train sample count: %d, eval sample count: %d' % (len(cur_x_train), len(cur_x_eval)))
        train_big_tag_model(big_tag, cur_x_train, cur_y_train, cur_x_eval, cur_y_eval)


class ModelManager(object):
    ALL_MODELS = {}

    @classmethod
    def load_the_whole_model(cls):
        print (u'loading the whole model')
        models = {u'all_big_tags': get_model_by_big_tag(u'all_big_tags')}
        models[u'all_big_tags'].load_weights(get_big_tag_model_save_path(u'all_big_tags'))
        for big_tag, big_tag_seq in refined_preprocessing.TagManager.BIG_TAG_TO_SEQ.items():
            if refined_preprocessing.TagManager.SUBTAG_COUNT[big_tag_seq] == 0:
                continue
            print (u'loading model of big tag: %s' % big_tag)
            models[big_tag] = get_model_by_big_tag(big_tag)
            models[big_tag].load_weights(get_big_tag_model_save_path(big_tag))
        cls.ALL_MODELS = models
        print (u'loading the whole model done..')

    @classmethod
    def get_predict_result(cls, big_tag, x):
        return cls.ALL_MODELS[big_tag].predict(x)


class Node(object):
    def __init__(self, x, start_tag=None, start_estimated_cost=0.0):
        self.x = x
        self.path = [start_tag]
        # start_estimated_cost = K.epsilon() if start_estimated_cost < K.epsilon() else start_estimated_cost
        self.cost = start_estimated_cost
        self.search_end = False

    def __cmp__(self, other):
        return self.cost - other.cost

    def __unicode__(self):
        return u'cur_tag: %s, cost: %lf' % (self.path[-1], self.cost)

    def expand(self, predict_results=None):
        if self.search_end:
            return []
        else:
            predict_results = ModelManager.get_predict_result(self.path[-1], self.x) \
                if predict_results is None else predict_results
            expand_nodes = []
            for seq in range(len(predict_results)):
                p = predict_results[seq]
                p = K.epsilon() if p < K.epsilon() else p
                sub_tag = refined_preprocessing.TagManager.SEQ_TO_SUB_TAG[self.path[-1]][seq]
                new_node = Node(self.x)
                new_node.path = self.path[:]
                new_node.path.append(sub_tag)
                new_node.search_end = True if new_node.path[-1] not in refined_preprocessing.TagManager.SEQ_TO_SUB_TAG \
                    else False
                new_node.cost += 1.0 / p
                expand_nodes.append(new_node)
            return expand_nodes


def get_predict_save_path(tag):
    return os.path.join(REFINED_MODEL_WEIGHTS_PATH_ROOT, u'%s predicts.npy' % tag)


# x is a numpy array (samples, 24*64)
def predict_eval_data_based_on_a_star(x):
    # print (x.shape)
    # print (ModelManager.ALL_MODELS[u'all_big_tags'].predict(x).shape)
    predicts = {}
    if os.path.exists(get_predict_save_path(u'all_big_tags')):
        predicts[u'all_big_tags'] = numpy.load(get_predict_save_path(u'all_big_tags'))
        for big_tag, big_tag_seq in refined_preprocessing.TagManager.BIG_TAG_TO_SEQ.items():
            if refined_preprocessing.TagManager.SUBTAG_COUNT[big_tag_seq] == 0:
                continue
            print (u'loading predict of %s' % big_tag)
            predicts[big_tag] = numpy.load(get_predict_save_path(big_tag))
    else:
        ModelManager.load_the_whole_model()
        print (u'predict model: all_big_tags')
        predicts[u'all_big_tags'] = ModelManager.ALL_MODELS[u'all_big_tags'].predict(x)
        for big_tag, big_tag_seq in refined_preprocessing.TagManager.BIG_TAG_TO_SEQ.items():
            if refined_preprocessing.TagManager.SUBTAG_COUNT[big_tag_seq] == 0:
                continue
            print (u'predict model: %s' % big_tag)
            predicts[big_tag] = ModelManager.ALL_MODELS[big_tag].predict(x)
        for tag, predict in predicts.items():
            numpy.save(get_predict_save_path(tag), predict)
    pred_lists = []
    for i in range(len(x)):
        # a-star shortest path finding
        q = []
        heapq.heappush(q, Node(x[i], u'all_big_tags', 0.0))
        while len(q) > 0:
            front_node = heapq.heappop(q)
            if front_node.search_end:
                pred_lists.append([refined_preprocessing.TagManager.idx(tag) for tag in front_node.path[1:]])
                break
            for node in front_node.expand(predict_results=predicts[front_node.path[-1]][i]):
                heapq.heappush(q, node)
    return pred_lists


def read_refined_eval_y_for_evaluation(size=None):
    eval_y_lists = []
    cnt = 0
    for line in open(refined_preprocessing.REFINED_Y_EVAL_SP):
        eval_y_lists.append([int(pair.split(u',')[1]) for pair in line.split()])
        cnt += 1
        if size and cnt == size:
            break
    return eval_y_lists


def subset_evaluator(pred_list, true_list):
    """
    subset
    """
    return set(pred_list) == set(true_list)


def hamming_evaluator(pred_list, true_list):
    """
    hamming loss
    """
    pred_set = set(pred_list)
    true_set = set(true_list)
    return len(pred_set ^ true_set)


def accuracy_evaluator(pred_list, true_list):
    """
    accuracy
    """
    pred_set = set(pred_list)
    true_set = set(true_list)
    return len(pred_set & true_set) / float(len(pred_set | true_set))


def precision_evaluator(pred_list, true_list):
    """
    precision
    """
    pred_set = set(pred_list)
    true_set = set(true_list)
    return len(pred_set & true_set) / float(len(pred_set)) if len(pred_set) != 0 else 0.0


def recall_evaluator(pred_list, true_list):
    """
    recall
    """
    pred_set = set(pred_list)
    true_set = set(true_list)
    return len(pred_set & true_set) / float(len(true_set))


def big_tag_correctness_evaluator(pred_list, true_list):
    """
    big tag correctness
    """
    return pred_list[0] == true_list[0]


def evaluation_sp(evaluators, size=None):
    x_eval_sp = refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_EVAL_SP, size=size)
    print (u'x_eval_sp loaded..start predict')
    pred_lists = predict_eval_data_based_on_a_star(x_eval_sp)
    print (u'prediction done..')
    eval_y_lists = read_refined_eval_y_for_evaluation(size=size)
    for evaluator in evaluators:
        total_score = 0.0
        for i in range(len(pred_lists)):
            total_score += evaluator(pred_lists[i], eval_y_lists[i])
        total_score /= len(pred_lists)
        print (evaluator.__doc__)
        print (total_score)


if __name__ == '__main__':
    # print refined_preprocessing.TagManager.SEQ_TO_SUB_TAG
    # print refined_preprocessing.TagManager.SEQ_TO_BIG_TAG
    # print refined_preprocessing.TagManager.IDX_TO_REFINED_TAG
    if sys.argv[1] == u'train':
        train(from_scratch=bool(sys.argv[2] if len(sys.argv) >= 3 else False))
    elif sys.argv[1] == u'eval':
        evaluation_sp([subset_evaluator, hamming_evaluator, accuracy_evaluator,
                       precision_evaluator, recall_evaluator, big_tag_correctness_evaluator],
                      int(sys.argv[2]) if len(sys.argv) >= 3 else None)
