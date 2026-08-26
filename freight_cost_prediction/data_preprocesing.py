"""
Backward-compatibility alias for data_preprocessing.py.
"""

from .data_preprocessing import load_vendor_data, prepare_features, prepare_featres, split_data

__all__ = ["load_vendor_data", "prepare_features", "prepare_featres", "split_data"]