from covid19_spread import load
import pandas as pd
import pytest


DATA_PATH_US_CSV = "covid19_spread/data/usa/data_cases.csv"
DATA_PATH_NY_CSV = "covid19_spread/data/usa/data_cases_ny.csv"


class TestLoad:
    @pytest.mark.parametrize("path", [DATA_PATH_US_CSV, DATA_PATH_NY_CSV])
    def test_load_cases_by_region(self, path):
        """Confirms cases loaded are per region"""
        cases_df = load.load_confirmed_by_region(path)
        assert cases_df.index.name == "date"
        assert type(cases_df.index) == pd.core.indexes.datetimes.DatetimeIndex
        assert (cases_df >= 0).all().all()

        regions = cases_df.columns
        suffolk_present = (
            "Suffolk County" in regions or "Suffolk County, New York" in regions
        )
        assert suffolk_present

    @pytest.mark.parametrize("path", [DATA_PATH_US_CSV, DATA_PATH_NY_CSV])
    def test_load_confirmed(self, path):
        df = load.load_confirmed(path, None)
        assert df.index.name == "date"
        assert (df >= 0).all()
        # should only have one column for total cases
        assert len(df.shape) == 1
