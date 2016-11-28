# coding = utf8
import lstm_with_tag_relation as my_model
import preprocessing
import numpy


def sample_based_validation(x_eval, y_eval, trained_model, threshold):
    y_pred = trained_model.predict(x_eval)
    y_pred *= y_pred >= threshold
    numpy.ceil(y_pred, y_pred)
    numpy.ceil(y_eval, y_eval)
    symmetric_diff = numpy.abs(y_pred - y_eval)
    print (numpy.sum(numpy.sum(symmetric_diff, axis=-1).clip(0.0, 1.0)) / x_eval.shape[0])
    print (u'-------hamming loss---------')
    print (numpy.sum(symmetric_diff) / x_eval.shape[0])


def subset_evaluator(y_pred, y_true):
    """
    subset evaluator
    """
    return my_model.derive_tag_indices_from_y(y_pred) == my_model.derive_tag_indices_from_y(y_true, is_y_true=True)


def hamming_evaluator(y_pred, y_true):
    """
    hamming loss
    """
    return len(my_model.derive_tag_indices_from_y(y_true) ^ my_model.derive_tag_indices_from_y(y_pred))


def evaluation(model_weights_path, evaluators):
    model = my_model.new_model()
    model.load_weights(model_weights_path)
    x_eval, y_eval = \
        preprocessing.read_x(preprocessing.X_EVAL_PATH), preprocessing.read_y(preprocessing.Y_EVAL_PATH)
    print (u'eval data loaded')
    y_pred = model.predict(x_eval)
    print (u'all prediction done')
    for evaluator in evaluators:
        total_score = 0.0
        for idx in range(len(y_pred)):
            total_score += evaluator(y_pred[idx], y_eval[idx])
        total_score /= len(y_pred)
        print (evaluator.__doc__)
        print (total_score)


def batch_evaluation(model_weights_path, batch_size):
    model = my_model.new_model()
    model.load_weights(model_weights_path)
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


def print_relation(model_weights_path):
    model = my_model.new_model()
    model.load_weights(model_weights_path)
    print (model.get_layer(u'relation').get_weights())


UNITNORM_MODEL = ur'../models/model_weights_unitnorm.h5'

if __name__ == '__main__':
    evaluation(UNITNORM_MODEL, [subset_evaluator, hamming_evaluator])
