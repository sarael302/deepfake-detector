# src/utils.py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc




def plot_history(history, out_path=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend(); plt.title('Accuracy')


    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend(); plt.title('Loss')


    if out_path:
        plt.savefig(out_path)
        plt.show()




def evaluate_and_report(model, dataset, class_names=['Fake','Real']):
    y_true = []
    y_pred = []
    y_prob = []
    for images, labels in dataset:
        probs = model.predict(images)
        preds = (probs >= 0.5).astype(int).flatten()
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.flatten().tolist())
        y_true.extend(labels.numpy().astype(int).tolist())
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC: {roc_auc:.4f}')
    return cm, y_true, y_pred, y_prob