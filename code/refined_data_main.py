# coding=utf8
from lstm_with_tag_relation import *
from evaluation import *


def print_usage():
    print (u'usage: python main.py train for training\n'
           u'       python main.py eval for evaluation')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
    elif sys.argv[1] == u'train':
        train_on_refined_data()
    elif sys.argv[1] == u'eval':
        evaluation_on_refined_data(sys.argv[2],
                                   [subset_evaluator, hamming_evaluator, accuracy_evaluator, precision_evaluator,
                                    recall_evaluator,
                                    one_error_evaluator, coverage_evaluator])
    elif sys.argv[1] == u'p_relation':
        print_relation(sys.argv[2])
    elif sys.argv[1] == u'lsq':
        lsq(sys.argv[2], sample_size=int(sys.argv[3]) if len(sys.argv) >= 4 else None, on_refined_data=True)
    else:
        print_usage()
