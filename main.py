from src.data_preprocessing import preprocess_data
from src.train_model import train_models
from src.evaluate_model import evaluate_model

from src.visualize import (
    plot_confusion_matrix,
    plot_accuracy_comparison
)


def main():

    print("Loading Dataset...")

    X_train, X_test, y_train, y_test = preprocess_data()

    print("\nTraining Models...")

    model, accuracies = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nEvaluating Best Model...")

    evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nDisplaying Accuracy Comparison Graph...")

    plot_accuracy_comparison(
        accuracies
    )

    print("\nDisplaying Confusion Matrix...")

    plot_confusion_matrix(
        model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()