# coding=utf8
import sys
from lstm_with_tag_relation import *
from evaluation import *


def print_usage():
    print (u'usage: python main.py train for training\n'
           u'       python main.py eval for evaluation')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
    elif sys.argv[1] == u'train':
        train()
    elif sys.argv[1] == u'eval':
        evaluation(sys.argv[2],
                   [subset_evaluator, hamming_evaluator, accuracy_evaluator, precision_evaluator, recall_evaluator,
                    one_error_evaluator, coverage_evaluator, ranking_loss_evaluator])
    elif sys.argv[1] == u'batch_eval':
        batch_evaluation(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == u'p_relation':
        print_relation(sys.argv[2])
    elif sys.argv[1] == u'lsq':
        lsq(sys.argv[2])
    else:
        print_usage()
