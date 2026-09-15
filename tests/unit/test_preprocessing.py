import pytest
import pandas as pd
from testbook import testbook


@pytest.fixture(scope="module")
def tb():
    # Execute notebook up to preprocessing definitions
    with testbook('notebooks/model_training.ipynb', execute=True, timeout=300) as tb:
        yield tb


def test_flatten_record(tb):
    flatten_record = tb.ref("flatten_record")

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


def test_flatten_record_missing_fields(tb):
    flatten_record = tb.ref("flatten_record")

    result = flatten_record({})

    assert result['age'] == 0
    assert result['city'] == "Unknown"
    assert result['target'] == 0


def test_preprocess_records(tb):
    tb.inject(
        "dummy_inputs = [{'customer': {'age': 25}}, {'customer': {'age': 40}}]\n"
        "df_test = preprocess_records(dummy_inputs)"
    )

    assert tb.value("len(df_test)") == 2
    assert tb.value("int(df_test.iloc[0]['age'])") == 25
    assert tb.value("int(df_test.iloc[1]['age'])") == 40
