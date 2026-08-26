"""
Inference module for Invoice Risk Flagging.
Loads the trained classifier and scaler to predict whether invoices require manual approval.
"""

from pathlib import Path
import joblib
import pandas as pd

# Paths to trained model and feature scaler artifacts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = PROJECT_ROOT / "invoice_flagging" / "models" / "scaler.pkl"

# Expected input feature columns
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_artifacts(model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
    """
    Load the trained classifier model and feature StandardScaler.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return model, scaler


def predict_invoice_flag(input_data) -> pd.DataFrame:
    """
    Predict risk flags (0 = Safe / Auto-Approve, 1 = Risk / Manual Approval) for invoices.

    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Must contain columns: invoice_quantity, invoice_dollars, Freight,
        total_item_quantity, total_item_dollars.

    Returns:
    --------
    pd.DataFrame: Original DataFrame enriched with 'Predicted_Flag'.
    """
    model, scaler = load_artifacts()
    input_df = pd.DataFrame(input_data)
    
    # Scale numerical features prior to classification
    features_scaled = scaler.transform(input_df[FEATURES])
    input_df['Predicted_Flag'] = model.predict(features_scaled)
    return input_df


def main():
    # Example inference run for local verification
    sample_data = {
        "invoice_quantity": [10, 500],
        "invoice_dollars": [150.0, 12000.0],
        "Freight": [5.0, 350.0],
        "total_item_quantity": [10, 480],
        "total_item_dollars": [150.0, 11500.0],
    }
    prediction = predict_invoice_flag(sample_data)
    print("Invoice Risk Prediction Results:\n", prediction)


if __name__ == "__main__":
    main()