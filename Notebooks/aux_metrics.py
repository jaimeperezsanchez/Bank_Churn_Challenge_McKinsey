import pandas as pd
import numpy as np

# Auxiliar functions
def get_uplift(y_true, y_score, N=100):
    '''
    Compute uplift table
    :param y_true: labels
    :param y_score: scores
    :param N: number of buckets
    :param plot: if not none, string containing name of the plot
    :return: pd.DataFrame with several metrics
    '''
    uplift_df = pd.DataFrame({"y_true":np.reshape(y_true, (-1)),
                              "y_score":np.reshape(y_score, (-1))})

    uplift_df = uplift_df.sort_values(by=["y_score"], ascending=False)
    uplift_df["rank"] = 1+np.arange(uplift_df.shape[0])
    uplift_df["n_tile"] = np.ceil(N* uplift_df["rank"] / uplift_df.shape[0])

    uplift_result = uplift_df.groupby(["n_tile"])["y_true"].agg({'num_observations':'count','num_positives':"sum", "segment_precision":"mean"})
    uplift_result['acc_observations'] = uplift_result['num_observations'].cumsum()
    uplift_result['acc_positives'] = uplift_result['num_positives'].cumsum()
    uplift_result['acc_precision'] = uplift_result['acc_positives'] / uplift_result['acc_observations']
    uplift_result['segment_uplift'] = uplift_result['segment_precision'] / np.mean(y_score)
    uplift_result['acc_uplift'] = uplift_result['acc_precision'] / np.mean(y_true)
    uplift_result['percentile'] = np.arange(1,101,step=100/N, dtype=int)
    uplift_result['bin'] = np.arange(1, N+1, dtype=int)
    uplift_result['segment_recall'] = uplift_result['num_positives'] / np.sum(y_true)
    uplift_result['acc_recall'] = uplift_result['acc_positives'] / np.sum(y_true)

    cols_sel = ["bin", "percentile", "acc_observations", "acc_positives", "acc_precision", "acc_recall", 
                  "acc_uplift", "num_observations", "segment_precision", "segment_recall", "segment_uplift"]
    return uplift_result[cols_sel]

def get_feature_importance(booster):
    feats = list(zip(
        booster.feature_name(),
        booster.feature_importance(importance_type='gain'),
        booster.feature_importance(importance_type='split')
    ))
    feats = pd.DataFrame(feats, columns=['column', 'importance_gain', 'importance_split'])
    feats['importance_gain']  = feats['importance_gain'] / feats['importance_gain'].sum()
    feats['importance_split'] = feats['importance_split'] / feats['importance_split'].sum()
    feats = feats.sort_values(by='importance_gain', ascending=False).reset_index(drop=True)
    return feats