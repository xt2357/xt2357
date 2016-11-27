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
        evaluation(float(sys.argv[2]))
    elif sys.argv[1] == u'batch_eval':
        batch_evaluation(int(sys.argv[2]))
    elif sys.argv[1] == u'p_relation':
        print_relation()
    else:
        print_usage()
