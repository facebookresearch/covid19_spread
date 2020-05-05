import load
import pandas as pd


DATA_PATH_CSV = "data/usa/data.csv"
DATA_PATH_H5 = "data/nystate/timeseries.h5"
POP_PATH = "data/population-data/US-states/new-york-population.csv"

class TestLoad:
    def test_load_populations_by_region(self):
        """Verifies populations match regions in length"""
        df = load.load_populations_by_region(POP_PATH)
        assert isinstance(df, pd.DataFrame)
        assert df["region"].shape[0] == df["population"].shape[0]

    def test_load_cases_by_region_h5(self):
        """Confirms cases loaded are per region"""
        cases_df = load.load_confirmed_by_region(DATA_PATH_H5)
        assert cases_df.index.name == "date"
        assert (cases_df >= 0).all().all()
        assert "Suffolk" in cases_df.columns

    def test_load_confirmed_by_region_csv(self):
        cases_df = load.load_confirmed_by_region(DATA_PATH_CSV)
        print(cases_df)
        assert cases_df.index.name == "date"
        assert (cases_df >= 0).all().all()

    def test_regions_match_in_cases_and_population(self):
        """Verifies the regions in cases and population data match"""
        cases_df = load.load_confirmed_by_region(DATA_PATH_H5)
        populations_df = load.load_populations_by_region(POP_PATH)
        case_regions = sorted(cases_df.columns)
        population_regions = sorted(populations_df["region"].tolist())
        assert case_regions == population_regions

    def test_load_confirmed_h5(self):
        populations_df = load.load_populations_by_region(POP_PATH)
        regions = populations_df["region"].tolist()
        df = load.load_confirmed(DATA_PATH_H5, regions)
        assert df.index.name == "date"
        assert (df >= 0).all()
        # should only have one column for total cases
        assert len(df.shape) == 1


