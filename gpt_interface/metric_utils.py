from sklearn.metrics import precision_recall_fscore_support
import krippendorff
from scipy.stats import bootstrap
import pandas as pd

class sym:

    def __call__(self):
        return [
            v for k, v in self.__class__.__dict__.items()
            if not k.startswith('__')
        ]

    def __iter__(self):
        return iter(self())

    def __contains__(self, item):
        return item in self()

class stat(sym):
    kripp_alpha = "Krippendorff's alpha"
    ci_low = "CI low"
    ci_high = "CI high"
    n = 'n'

def bootstrap_ci(data, statistic_fn, n_resamples=10**3, confidence_level=0.95):
    wrapped_data = [dict(point=d) for d in data]
    statistic_fn_wrapper = lambda ds: statistic_fn([d['point'] for d in ds])
    result = bootstrap((wrapped_data,), statistic_fn_wrapper, vectorized=False,
                                n_resamples=n_resamples, confidence_level=confidence_level)
    return result.confidence_interval

def krippendorfs_alpha(df, ci=True, level='ordinal'):
    """
    :param df: pandas dataframe: items x labeler: label
    :return:
    """
    ratings = df.to_numpy()
    ka = lambda x: krippendorff.alpha(x.T, level_of_measurement=level)
    try:
        alpha = ka(ratings)
    except AssertionError:
        alpha = None
    if ci:
        try:
            low, high = bootstrap_ci(ratings, lambda x: ka(np.array(x)))
        except AssertionError:
            low, high = None, None
        result = {
            stat.kripp_alpha: alpha,
            stat.ci_low: low, stat.ci_high: high,
            stat.n: len(df)
        }
    else:
        result = {
            stat.kripp_alpha: alpha,
            stat.n: len(df)
        }
    return result["Krippendorff's alpha"]

def get_metrics(pred_labels, true_labels, do_print=True):
    assert len(pred_labels) == len(true_labels)
    p, r, f, _ = precision_recall_fscore_support(
        y_true=true_labels,
        y_pred=pred_labels,
        average='binary',
        pos_label=1,
        zero_division=0
    )
    a = sum([1 for i in range(len(pred_labels)) if pred_labels[i] == true_labels[i]]) / len(pred_labels)
    # this is the same as macro-recall!!!
    # acc_pos = sum([1 for i in range(len(pred_labels)) if
    #                pred_labels[i] == true_labels[i] and true_labels[i] == 1]) / true_labels.count(1)
    # acc_neg = sum([1 for i in range(len(pred_labels)) if
    #                pred_labels[i] == true_labels[i] and true_labels[i] == 0]) / true_labels.count(
    #     0) if true_labels.count(0) > 0 else 0
    # macro_acc = (acc_pos + acc_neg) / 2
    s = true_labels.count(1)
    s_m = true_labels.count(0)
    t = len(pred_labels)
    df = pd.DataFrame.from_dict({'pred': pred_labels, 'true': true_labels}, orient='index').T
    alpha = krippendorfs_alpha(df, ci=False, level='nominal')
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true=true_labels,
        y_pred=pred_labels,
        average='binary',
        pos_label=0,
        zero_division=0
    )
    if do_print:
        print(f"\tP: {p:.2f} | R: {r:.2f} | F1: {f:.2f} [{s}] | P-m: {p:.2f} | R-m: {r:.2f} | F1-m: {f:.2f} | A: {a:.2f} | Alpha: {alpha:.2f} [{t}]")
    return p, r, f, s, p_m, r_m, f_m, s_m, a, alpha, t