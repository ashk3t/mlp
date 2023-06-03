import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    auc,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def get_data():
    data_folder = "data"
    header = open(data_folder + "/agaricus-lepiota.header").read()[:-1].split(",")
    df = pd.read_table(
        data_folder + "/agaricus-lepiota.data",
        delimiter=",",
        names=["class"] + header,
        index_col=False,
    )
    header = [f for f in header if df[f].unique().shape[0] > 1]
    df = df.loc[:, header + ["class"]]
    X, y = df.loc[:, header], df["class"]
    return X, y


def prepare_data(X, y):
    onehot_encoder = preprocessing.OneHotEncoder()
    label_encoder = preprocessing.LabelEncoder()

    X = onehot_encoder.fit_transform(X)
    y = label_encoder.fit_transform(y)
    pos_label = list(label_encoder.classes_).index("e")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    return X_train, X_test, y_train, y_test, pos_label


def show_results(y_test, y_pred, y_proba, pos_label):
    def draw_roc_auc(y_test, y_proba, pos_label):
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"Target ROC (area = {roc_auc : 0.4f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-AUC")
        plt.legend(loc="lower right")
        plt.show()

    print("accuracy:", accuracy_score(y_test, y_pred))
    print("precision:", precision_score(y_test, y_pred, pos_label=pos_label))
    print("recall:", recall_score(y_test, y_pred, pos_label=pos_label))
    print("f1:", f1_score(y_test, y_pred, pos_label=pos_label))
    draw_roc_auc(y_test, y_proba[:, pos_label], pos_label)
