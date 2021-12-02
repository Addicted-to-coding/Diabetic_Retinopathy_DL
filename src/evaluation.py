from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def class_report(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    return class_report_df

def test_evaluate(model, objective, loader):
    model.eval()
    total_loss = 0
    size = 0
    
    y_prob = []
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(loader):

            X, y = data_batch[0].to(device).float(), data_batch[1].to(device)
            # X, y = map(lambda t: t.to(device).float(), (X, y))

            prediction = model(X)
            total_loss += objective(prediction, y) * X.shape[0]
            
            pred = softmax(prediction)            
            pred = np.array([np.argmax(x) for x in pred])
            
            y_pred = np.append(y_pred, pred)
            y_true = np.append(y_true, y.numpy())

            size += X.shape[0]
    
    report = class_report(y_true, y_pred)
    print(report)

    total_loss = total_loss / size
    return total_loss