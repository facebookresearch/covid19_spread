from .process_mobility import main as mobility_main
from .process_open_data import main as open_data_main


def prepare():
    mobility_main()
    open_data_main()
