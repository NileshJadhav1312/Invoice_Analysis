"""
Data preprocessing and feature engineering for Invoice Risk Flagging.
Joins purchase order line items with vendor invoice headers from SQLite,
creates ground-truth risk labels, and scales numerical features.
"""

from pathlib import Path
import sqlite3
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_default_db_path() -> str:
    """
    Locate the SQLite database across standard project directories.
    """
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir / "data" / "inventory.db",
        current_dir.parent / "data" / "inventory.db",
        current_dir.parent / "notebooks" / "inventory.db",
        Path("data/inventory.db").resolve(),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "data/inventory.db"


def load_invoice_data(db_path: str = None) -> pd.DataFrame:
    """
    Query and aggregate purchase orders and vendor invoices from SQLite.

    Parameters:
    -----------
    db_path : str, optional
        Path to inventory.db SQLite database.

    Returns:
    --------
    pd.DataFrame: Merged dataset containing invoice and PO aggregation features.
    """
    if db_path is None:
        db_path = get_default_db_path()

    with sqlite3.connect(db_path) as conn:
        query = """
        WITH purchase_agg AS (
            SELECT
                p.PONumber,
                COUNT(DISTINCT p.Brand) AS total_brands,
                SUM(p.Quantity) AS total_item_quantity,
                SUM(p.Dollars) AS total_item_dollars,
                AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
            FROM purchases p
            GROUP BY p.PONumber
        )
        SELECT
            vi.PONumber,
            vi.Quantity AS invoice_quantity,
            vi.Dollars AS invoice_dollars,
            vi.Freight,
            (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
            (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
            pa.total_brands,
            pa.total_item_quantity,
            pa.total_item_dollars,
            pa.avg_receiving_delay
        FROM vendor_invoice vi
        LEFT JOIN purchase_agg pa
            ON vi.PONumber = pa.PONumber
        """
        df = pd.read_sql_query(query, conn)
    return df


def create_invoice_risk_label(row: pd.Series) -> int:
    """
    Assign binary risk label based on invoice discrepancies:
    - Flag 1: Absolute price mismatch > $5 OR receiving delay > 10 days
    - Flag 0: Normal invoice
    """
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1
    if row["avg_receiving_delay"] > 10:
        return 1
    return 0


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply risk labeling rule to produce the 'flag_invoice' target column.
    """
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


def split_data(df: pd.DataFrame, features: list, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified train-test split preserving class distribution.
    """
    X = df[features]
    y = df[target]
    
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_features(X_train, X_test, scaler_path: str):
    """
    Standardize features using StandardScaler and save fitted scaler artifact.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_file = Path(scaler_path)
    scaler_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_file)
    
    return X_train_scaled, X_test_scaled
