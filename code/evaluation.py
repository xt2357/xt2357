# coding = utf8
import lstm_with_tag_relation as my_model
import nlp_utils
import preprocessing
import numpy


def text_predict(trained_model, text, threshold):
    # replace all the \n to make the whole text a single paragraph
    structure = nlp_utils.split_into_paragraph_sentence_token(text.replace(u'\n', u''))
    sentences = structure[0]
    input_v = preprocessing.padding_document(sentences)
    output_v = trained_model.predict([input_v])[0]
    output_v *= output_v > threshold
    idx = 0
    tags = []
    for confidence in output_v:
        if confidence > 0.0:
            tags.append((idx, confidence))
        idx += 1
    return [(preprocessing.TAG_IDX_TO_NAME[idx], confidence) for idx, confidence in tags]


def sample_based_validation(x_eval, y_eval, trained_model, threshold):
    y_pred = trained_model.predict(x_eval)
    y_pred *= y_pred >= threshold
    numpy.ceil(y_pred, y_pred)
    numpy.ceil(y_eval, y_eval)
    symmetric_diff = numpy.abs(y_pred - y_eval)
    print (numpy.sum(numpy.sum(symmetric_diff, axis=-1).clip(0.0, 1.0)) / x_eval.shape[0])
    print (u'-------hamming loss---------')
    print (numpy.sum(symmetric_diff) / x_eval.shape[0])


def evaluation(threshold):
    model = my_model.new_model()
    model.load_weights(my_model.MODEL_WEIGHTS_PATH)
    x_eval, y_eval = \
        preprocessing.read_x(preprocessing.X_EVAL_PATH), preprocessing.read_y(preprocessing.Y_EVAL_PATH)
    print (u'eval data loaded')
    sample_based_validation(x_eval, y_eval, model, threshold)


def batch_evaluation(batch_size):
    model = my_model.new_model()
    model.load_weights(my_model.MODEL_WEIGHTS_PATH)
    x_eval = preprocessing.read_x(preprocessing.X_EVAL_PATH, size=batch_size)
    y = model.predict(x_eval)
    with open('../data/nyt/batch_y.txt', 'w') as f:
        for output_v in y:
            idx = 0
            tags = []
            for confidence in output_v:
                if confidence > 0.0:
                    tags.append((idx, confidence))
                idx += 1
            tags.sort(lambda a, b: cmp(a[1], b[1]), reverse=True)
            f.write(str(tags))
            f.write('\n')


def print_relation():
    model = my_model.new_model()
    model.load_weights(my_model.MODEL_WEIGHTS_PATH)
    print (model.get_layer(u'relation').get_weights())


if __name__ == '__main__':
    pass
