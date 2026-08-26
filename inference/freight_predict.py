"""
Inference module for Freight Cost Prediction.
Loads the trained regression model to estimate shipping/freight cost for input invoices.
"""

from pathlib import Path
import joblib
import pandas as pd

# Paths to model artifacts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "freight_cost_prediction" / "models" / "predict_freight_cost_model.pkl"


def load_model(model_path: Path = MODEL_PATH):
    """
    Load the serialized freight cost prediction model from disk.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data) -> pd.DataFrame:
    """
    Predict freight cost for new vendor invoices.

    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Invoice data containing at least the 'Dollars' column.

    Returns:
    --------
    pd.DataFrame: Original DataFrame enriched with 'Predicted_Freight'.
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Freight'] = model.predict(input_df[['Dollars']]).round(2)
    return input_df


if __name__ == "__main__":
    # Example inference run for local testing
    sample_data = {
        "Dollars": [18500, 9000]
    }
    prediction = predict_freight_cost(sample_data)
    print("Freight Prediction Results:\n", prediction)