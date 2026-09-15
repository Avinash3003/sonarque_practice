import pytest
import pandas as pd
from src.preprocessing import flatten_record, preprocess_records


class TestFlattenRecord:
    """Unit tests for the flatten_record function."""

    def test_flatten_record_basic(self):
        """Test flattening a complete, valid nested record."""
        dummy_input = {
            "customer": {
                "age": 30,
                "location": {"city": "Seattle", "country": "USA"}
            },
            "activity": {"sessions": 10, "avg_duration": 20.5},
            "purchase": {"amount": 100, "previous_orders": 2},
            "target": 1
        }
        result = flatten_record(dummy_input)

        assert result['age'] == 30
        assert result['city'] == "Seattle"
        assert result['country'] == "USA"
        assert result['sessions'] == 10
        assert result['avg_duration'] == 20.5
        assert result['previous_orders'] == 2
        assert result['target'] == 1

    def test_flatten_record_missing_fields(self):
        """Test that missing fields fall back to defaults gracefully."""
        result = flatten_record({})

        assert result['age'] == 0
        assert result['city'] == "Unknown"
        assert result['country'] == "Unknown"
        assert result['sessions'] == 0
        assert result['avg_duration'] == 0.0
        assert result['previous_orders'] == 0
        assert result['target'] == 0

    def test_flatten_record_partial_data(self):
        """Test flattening a record with only some fields present."""
        partial_input = {
            "customer": {"age": 25},
            "target": 1
        }
        result = flatten_record(partial_input)

        assert result['age'] == 25
        assert result['city'] == "Unknown"
        assert result['target'] == 1


class TestPreprocessRecords:
    """Unit tests for the preprocess_records function."""

    def test_preprocess_records_returns_dataframe(self):
        """Test that the output is a pandas DataFrame."""
        records = [{"customer": {"age": 25}}, {"customer": {"age": 40}}]
        df = preprocess_records(records)

        assert isinstance(df, pd.DataFrame)

    def test_preprocess_records_row_count(self):
        """Test that one row is produced per record."""
        records = [{"customer": {"age": 25}}, {"customer": {"age": 40}}]
        df = preprocess_records(records)

        assert len(df) == 2

    def test_preprocess_records_correct_values(self):
        """Test that values are correctly mapped into DataFrame rows."""
        records = [{"customer": {"age": 25}}, {"customer": {"age": 40}}]
        df = preprocess_records(records)

        assert int(df.iloc[0]['age']) == 25
        assert int(df.iloc[1]['age']) == 40

    def test_preprocess_records_empty_input(self):
        """Test that an empty list returns an empty DataFrame."""
        df = preprocess_records([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
