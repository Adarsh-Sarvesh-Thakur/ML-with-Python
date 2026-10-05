from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]

y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]

target_class = 1

precision = precision_score(
    y_true,
    y_pred,
    pos_label=target_class
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=target_class
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=target_class
)

print("Evaluation Metrics for Target Class:", target_class)
print("------------------------------------------")
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)
