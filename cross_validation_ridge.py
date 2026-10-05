from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
import numpy as np

data = load_diabetes()

X = data.data
y = data.target

model = Ridge(alpha=1.0)

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="r2"
)

print("R² Scores for Each Fold:")
print(scores)

mean_score = np.mean(scores)

std_score = np.std(scores)

print("\nMean R² Score:", mean_score)
print("Standard Deviation:", std_score)
