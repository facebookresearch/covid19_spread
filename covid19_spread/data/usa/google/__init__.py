from .process_mobility import main as mobility_main
from .process_open_data import main as open_data_main


def prepare(resolution):
    mobility_main()
    open_data_main()
