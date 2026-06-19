import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

def evaluate_model(y_test, y_pred, y_proba, class_names):
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred)

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')

    return report, cm, roc_auc

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues', ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def plot_roc_auc(y_test, y_proba, class_names):
    from sklearn.metrics import roc_curve, auc
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{class_name} (AUC={auc_score:.2f})')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-AUC Curve')
    ax.legend()
    return fig