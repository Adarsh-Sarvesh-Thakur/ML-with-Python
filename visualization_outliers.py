import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def visualize_data(df, column_name):

    if column_name not in df.columns:
        print("Column not found in the DataFrame.")
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[column_name])
    plt.title("Boxplot")
    plt.ylabel(column_name)

    plt.subplot(1, 2, 2)
    plt.scatter(df.index, df[column_name])
    plt.title("Scatter Plot")
    plt.xlabel("Index")
    plt.ylabel(column_name)

    plt.tight_layout()
    plt.show()

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("Dataset:")
print(df.head())

visualize_data(df, "sepal length (cm)")
