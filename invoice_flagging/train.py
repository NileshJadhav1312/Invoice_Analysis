"""
Training pipeline for Invoice Risk Classification.
Loads merged invoice and PO records, generates risk flags, scales features,
tunes a Random Forest Classifier using GridSearchCV, and exports model artifacts.
"""

from pathlib import Path
import joblib
from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from model_evaluation import train_random_forest, evaluate_classifier

# Top discriminative features identified during exploratory analysis
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

TARGET = "flag_invoice"


def main():
    # 1. Setup output directory
    project_dir = Path(__file__).resolve().parent
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = models_dir / "scaler.pkl"
    model_path = models_dir / "predict_flag_invoice.pkl"

    # 2. Load and label data
    print("Loading and labeling invoice data...")
    df = load_invoice_data()
    df = apply_labels(df)

    # 3. Stratified split and scale features
    print("Splitting and scaling features...")
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, str(scaler_path)
    )

    # 4. Train and optimize Random Forest model with GridSearch
    print("Training Random Forest model with GridSearch (5-fold CV)...")
    grid_search = train_random_forest(X_train_scaled, y_train)

    # 5. Evaluate best estimator on holdout test set
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier",
    )

    # 6. Save model and scaler artifacts
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Model successfully saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    main()