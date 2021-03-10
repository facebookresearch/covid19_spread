from .fetch import main as fetch, SIGNALS
from .process_symptom_survey import main as process


def prepare(resolution):
    for source, signal in SIGNALS:
        fetch(resolution, source, signal)
        process(f"{source}/{signal}", resolution)
