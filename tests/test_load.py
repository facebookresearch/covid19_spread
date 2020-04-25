import load


DATA_PATH = "timeseries_filtered.h5"
POP_PATH = "data/population-data/US-states/new-jersey-population.csv"



class TestLoad:

    def test_load_populations_by_region(self):
        """Verifies populations match regions in length"""
        populations, regions = load.load_populations_by_region(POP_PATH)
        assert isinstance(populations, list)
        assert len(regions) == len(populations)

    def test_load_populations_by_region_filter(self):
        """Verifies populations match regions in length"""
        _, regions = load.load_populations_by_region(POP_PATH)
        nodes = regions[:3]
        populations, regions = load.load_populations_by_region(POP_PATH, nodes=nodes)
        assert len(populations) == len(nodes)

    def test_load_cases_by_region(self):
        """Confirms cases loaded are per region"""
        _, regions = load.load_populations_by_region(POP_PATH)
        cases_by_region, _, _ = load.load_confirmed_by_region(DATA_PATH)
        assert cases_by_region.shape[0] == len(regions)
        # confirm length of cases per region is correct
        cases = load.load_confirmed(DATA_PATH, regions)
        assert cases_by_region[0].shape == cases.shape