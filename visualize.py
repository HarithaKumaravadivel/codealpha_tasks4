import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
        model,
        X_test,
        y_test):

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title(
        "Heart Disease Prediction - Confusion Matrix"
    )

    plt.show()


def plot_accuracy_comparison(results):

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        results.keys(),
        results.values()
    )

    plt.ylabel("Accuracy")
    plt.xlabel("Models")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)

    plt.show()