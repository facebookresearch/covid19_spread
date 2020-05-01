import load
import pandas as pd


DATA_PATH = "timeseries_filtered.h5"
POP_PATH = "data/population-data/US-states/new-jersey-population.csv"



class TestLoad:

    def test_load_populations_by_region(self):
        """Verifies populations match regions in length"""
        df = load.load_populations_by_region(POP_PATH)
        assert isinstance(df, pd.DataFrame)
        assert df["region"].shape[0] == df["population"].shape[0]

    def test_load_cases_by_region(self):
        """Confirms cases loaded are per region"""
        populations_df = load.load_populations_by_region(POP_PATH)
        cases_by_region, _, _ = load.load_confirmed_by_region(DATA_PATH)
        assert cases_by_region.shape[0] == populations_df.shape[0]
        # confirm length of cases per region is correct
        cases = load.load_confirmed(DATA_PATH, populations_df["region"].values)
        assert cases_by_region[0].shape == cases.shape