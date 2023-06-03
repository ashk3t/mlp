from utils import get_data, prepare_data, show_results
from model import MLPClassifier


if __name__ == "__main__":
    X, y = get_data()
    X_train, X_test, y_train, y_test, pos_label = prepare_data(X, y)

    mlp = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation="logistic",
        learning_rate=0.01,
        max_iter=500
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_proba = mlp.predict_proba(X_test)

    show_results(y_test, y_pred, y_proba, pos_label)
