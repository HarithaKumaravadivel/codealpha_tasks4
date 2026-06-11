from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
import joblib
import os


def train_models(X_train, X_test, y_train, y_test):

    models = {

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "SVM":
        SVC(kernel='rbf'),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ),

        "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        )
    }

    accuracies = {}

    best_model = None
    best_accuracy = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        accuracies[name] = accuracy

        print(f"\n{name} Accuracy = {accuracy:.4f}")

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print("\nBest Accuracy:", best_accuracy)

    return best_model, accuracies