# coding = utf8
import lstm_with_tag_relation as my_model
import preprocessing
import numpy
import sys
import os
import refined_preprocessing


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
    return len(my_model.derive_tag_indices_from_y(y_true, is_y_true=True) ^ my_model.derive_tag_indices_from_y(y_pred))


def accuracy_evaluator(y_pred, y_true):
    """
    accuracy
    """
    pred_set = my_model.derive_tag_indices_from_y(y_pred)
    true_set = my_model.derive_tag_indices_from_y(y_true, is_y_true=True)
    return len(pred_set & true_set) / float(len(pred_set | true_set))


def precision_evaluator(y_pred, y_true):
    """
    precision
    """
    pred_set = my_model.derive_tag_indices_from_y(y_pred)
    true_set = my_model.derive_tag_indices_from_y(y_true, is_y_true=True)
    # if len(pred_set) == 0:
    #     print (u'predict set has no element')
    return len(pred_set & true_set) / float(len(pred_set)) if len(pred_set) != 0 else 0.0


def recall_evaluator(y_pred, y_true):
    """
    recall
    """
    pred_set = my_model.derive_tag_indices_from_y(y_pred)
    true_set = my_model.derive_tag_indices_from_y(y_true, is_y_true=True)
    return len(pred_set & true_set) / float(len(true_set))


def one_error_evaluator(y_pred, y_true):
    """
    ranking: one error
    """
    true_set = my_model.derive_tag_indices_from_y(y_true, is_y_true=True)
    asc_idx = numpy.argsort(y_pred)
    return asc_idx[-1] not in true_set


def coverage_evaluator(y_pred, y_true):
    """
    ranking: coverage
    """
    true_set = my_model.derive_tag_indices_from_y(y_true, is_y_true=True)
    asc_idx = numpy.argsort(y_pred)
    cnt = 0
    for idx in range(len(asc_idx) - 1, -1, -1):
        if asc_idx[idx] in true_set:
            cnt += 1
        if len(true_set) == cnt:
            return len(asc_idx) - idx


def ranking_loss_evaluator(y_pred, y_true):
    """
    ranking: ranking loss
    """
    true_set = my_model.derive_tag_indices_from_y(y_true, is_y_true=True)
    others = set(range(preprocessing.MEANINGFUL_TAG_SIZE)) - true_set
    reverse_pair = 0.0
    for true in true_set:
        for untrue in others:
            if y_pred[true] <= y_pred[untrue]:
                reverse_pair += 1
    return reverse_pair / len(true_set) / len(others)


def big_tag_correctness_evaluator(y_pred, y_true):
    """
    big tag correctness
    """
    pred_set = my_model.derive_tag_indices_from_y(y_pred)
    true_set = my_model.derive_tag_indices_from_y(y_true, is_y_true=True)
    big_tag_idx = 0
    for tag_idx in true_set:
        if refined_preprocessing.TagManager.IDX_TO_REFINED_TAG[tag_idx].find(u'#') == -1:
            big_tag_idx = tag_idx
            break
    return big_tag_idx in pred_set


# def average_precision_evaluator(y_pred, y_true):
#     """
#     ranking: average precision
#     """
#     true_set = my_model.derive_tag_indices_from_y(y_true, is_y_true=True)
#     asc_idx = numpy.argsort(y_pred)
#     return 1


def evaluation(model_weights_path, evaluators):
    my_model.read_threshold_lsq_coefficient()
    if model_weights_path.endswith(u'hdf5'):
        import keras
        model = keras.models.load_model(model_weights_path)
    else:
        model = my_model.new_model()
        model.load_weights(model_weights_path)
    x_eval, y_eval = \
        preprocessing.read_x(preprocessing.X_EVAL_IGNORE_STOP_PATH), \
        preprocessing.read_y(preprocessing.Y_EVAL_IGNORE_STOP_PATH)
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


def evaluation_on_refined_data(model_weights_path, evaluators):
    import refined_preprocessing
    my_model.read_threshold_lsq_coefficient()
    if model_weights_path.endswith(u'hdf5'):
        import keras
        model = keras.models.load_model(model_weights_path)
    else:
        model = my_model.new_model(refined_preprocessing.TagManager.REFINED_TAG_DICT_SIZE)
        model.load_weights(model_weights_path)
    x_eval, y_eval = \
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_EVAL_SP), \
        refined_preprocessing.read_refined_y(refined_preprocessing.REFINED_Y_EVAL_SP, return_idx=True)
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


def evaluation_baseline(model_weights_path, evaluators):
    import refined_preprocessing
    my_model.read_threshold_lsq_coefficient()
    if model_weights_path.endswith(u'hdf5'):
        import keras
        model = keras.models.load_model(model_weights_path)
    else:
        model = my_model.new_model(refined_preprocessing.TagManager.REFINED_TAG_DICT_SIZE, baseline=True)
        model.load_weights(model_weights_path)
    x_eval, y_eval = \
        refined_preprocessing.read_refined_x(refined_preprocessing.REFINED_X_EVAL_SP), \
        refined_preprocessing.read_refined_y(refined_preprocessing.REFINED_Y_EVAL_SP, return_idx=True)
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
    x_eval = preprocessing.read_x(preprocessing.X_EVAL_IGNORE_STOP_PATH, size=batch_size)
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


UNITNORM_MODEL_PATH = os.path.join(os.path.dirname(__file__), ur'../models/model_weights_unitnorm.h5')

if __name__ == '__main__':
    evaluation(sys.argv[1],
               [subset_evaluator, hamming_evaluator, accuracy_evaluator, precision_evaluator, recall_evaluator,
                one_error_evaluator, coverage_evaluator, ranking_loss_evaluator])
