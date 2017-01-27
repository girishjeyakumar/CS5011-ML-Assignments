from sklearn import metrics

def print_classification_metrics(y_true, y_pred):
    print "Best fit accuracy: ", metrics.accuracy_score(y_true, y_pred)
    print "Precision: ", metrics.precision_score(y_true, y_pred)
    print "Recall: ", metrics.recall_score(y_true, y_pred)
    print "F-measure: ", metrics.f1_score(y_true, y_pred)
