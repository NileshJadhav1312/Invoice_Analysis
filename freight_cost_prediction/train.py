"""
Training pipeline for Freight Cost Prediction.
Loads vendor invoice data, trains regression candidates, benchmarks performance,
and serializes the best performing model to disk.
"""

from pathlib import Path
import joblib
from data_preprocessing import load_vendor_data, prepare_features, split_data
from model_evaluation import (
    train_LinearRegression,
    train_DecisionTree,
    train_RandomForest,
    evaluate_model,
)


def main():
    # 1. Resolve database path
    project_dir = Path(__file__).resolve().parent
    root_dir = project_dir.parent

    database_paths = [
        project_dir / "data" / "inventory.db",
        root_dir / "data" / "inventory.db",
        root_dir / "notebooks" / "inventory.db",
    ]

    db_path = next((path for path in database_paths if path.exists()), None)

    if db_path is None:
        searched_paths = "\n".join(str(path) for path in database_paths)
        raise FileNotFoundError(
            f"Database file not found. Searched:\n{searched_paths}"
        )

    # 2. Setup models directory
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load and prepare dataset
    print("Loading vendor invoice data...")
    df = load_vendor_data(str(db_path))
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Train regression model candidates
    print("Training candidate models (Linear Regression, Decision Tree, Random Forest)...")
    lr_model = train_LinearRegression(X_train, y_train)
    dt_model = train_DecisionTree(X_train, y_train)
    rf_model = train_RandomForest(X_train, y_train)

    # 5. Evaluate models on test dataset
    results = [
        evaluate_model(lr_model, X_test, y_test, "Linear Regression"),
        evaluate_model(dt_model, X_test, y_test, "Decision Tree"),
        evaluate_model(rf_model, X_test, y_test, "Random Forest"),
    ]

    # 6. Select and persist the best model (lowest MAE)
    best_result = min(results, key=lambda result: result["mae"])
    best_model_name = best_result["model_name"]

    model_map = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
    }
    best_model = model_map[best_model_name]

    model_path = models_dir / "predict_freight_cost_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"Best model '{best_model_name}' saved to {model_path}")


if __name__ == "__main__":
    main()