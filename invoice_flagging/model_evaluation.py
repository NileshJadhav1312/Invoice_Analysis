"""
Model training and evaluation module for Invoice Risk Classification.
Implements Random Forest hyperparameter tuning with GridSearchCV optimizing F1 score.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score


def train_random_forest(X_train, y_train):
    """
    Train and tune Random Forest Classifier via 5-fold GridSearchCV optimizing for F1 score.

    Parameters:
    -----------
    X_train : array-like
        Scaled training features.
    y_train : array-like
        Training target labels.

    Returns:
    --------
    GridSearchCV: Fitted grid search object containing best_estimator_.
    """
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 6],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini"],
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_classifier(model, X_test, y_test, model_name: str):
    """
    Evaluate classification model and display Accuracy and detailed Classification Report.
    """
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)