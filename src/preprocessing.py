"""
Preprocessing utilities for nested JSON records.
These functions are shared between the notebook and unit tests.
"""
import pandas as pd


def flatten_record(record: dict) -> dict:
    """
    Flatten a nested JSON record into a single-level dictionary.

    Args:
        record: Nested dict with keys: customer, activity, purchase, target

    Returns:
        Flat dictionary with all fields at the top level.
    """
    customer = record.get('customer', {})
    location = customer.get('location', {})
    activity = record.get('activity', {})
    purchase = record.get('purchase', {})

    return {
        'age': customer.get('age', 0),
        'city': location.get('city', 'Unknown'),
        'country': location.get('country', 'Unknown'),
        'sessions': activity.get('sessions', 0),
        'avg_duration': activity.get('avg_duration', 0.0),
        'previous_orders': purchase.get('previous_orders', 0),
        'target': record.get('target', 0)
    }


def preprocess_records(raw_records: list) -> pd.DataFrame:
    """
    Convert a list of nested JSON records into a flat pandas DataFrame.

    Args:
        raw_records: List of nested dicts

    Returns:
        Flat pandas DataFrame ready for model training.
    """
    flat_data = [flatten_record(r) for r in raw_records]
    return pd.DataFrame(flat_data)
