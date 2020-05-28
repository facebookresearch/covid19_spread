import load
import pandas as pd
import pytest


DATA_PATH_CSV = "data/usa/data_cases.csv"
DATA_PATH_H5 = "data/nystate/timeseries.h5"
POP_PATH = "data/population-data/US-states/new-york-population.csv"


class TestLoad:
    def test_load_populations_by_region(self):
        """Verifies populations match regions in length"""
        df = load.load_populations_by_region(POP_PATH)
        assert isinstance(df, pd.DataFrame)
        assert df["region"].shape[0] == df["population"].shape[0]

    @pytest.mark.parametrize("path", [DATA_PATH_CSV, DATA_PATH_H5])
    def test_load_cases_by_region(self, path):
        """Confirms cases loaded are per region"""
        cases_df = load.load_confirmed_by_region(path)
        assert cases_df.index.name == "date"
        assert type(cases_df.index) == pd.core.indexes.datetimes.DatetimeIndex
        assert (cases_df >= 0).all().all()

        regions = cases_df.columns
        suffolk_present = "Suffolk" in regions or "Suffolk, New York" in regions
        assert suffolk_present

    def test_regions_match_in_cases_and_population(self):
        """Verifies the regions in cases and population data match"""
        cases_df = load.load_confirmed_by_region(DATA_PATH_H5)
        populations_df = load.load_populations_by_region(POP_PATH)
        case_regions = sorted(cases_df.columns)
        population_regions = sorted(populations_df["region"].tolist())
        assert case_regions == population_regions

    @pytest.mark.parametrize("path", [DATA_PATH_CSV, DATA_PATH_H5])
    def test_load_confirmed(self, path):
        df = load.load_confirmed(path, None)
        assert df.index.name == "date"
        assert (df >= 0).all()
        # should only have one column for total cases
        assert len(df.shape) == 1
