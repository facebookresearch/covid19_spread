from covid19_spread.data.usa.symptom_survey.fetch import main as fetch, SIGNALS
from covid19_spread.data.usa.symptom_survey.process_symptom_survey import (
    main as process,
)
import os
import pandas

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def concat_mask_data(resolution):
    df1 = pandas.read_csv(
        os.path.join(SCRIPT_DIR, resolution, "fb-survey", "smoothed_wearing_mask.csv")
    )
    df2 = pandas.read_csv(
        os.path.join(
            SCRIPT_DIR, resolution, "fb-survey", "smoothed_wearing_mask_7d.csv"
        )
    )
    df2.columns = [c.replace("_7d", "") for c in df2.columns]
    df = pandas.concat([df1, df2])
    df.columns = [
        c.replace("smoothed_wearing_mask", "smoothed_wearing_mask_all")
        for c in df.columns
    ]
    df = df.sort_values(by="signal", ascending=False)
    df["signal"] = "smoothed_wearing_mask_all"
    df = df.drop_duplicates([resolution, "date"])
    df.to_csv(
        os.path.join(
            SCRIPT_DIR, resolution, "fb-survey", "smoothed_wearing_mask_all.csv"
        ),
        index=False,
    )


def prepare():
    for source, signal in SIGNALS:
        fetch("state", source, signal)
        fetch("county", source, signal)
    concat_mask_data("county")
    concat_mask_data("state")

    for source, signal in SIGNALS:
        if "wearing_mask" in signal:
            # Skip these since we end up concatenating the wearing_mask and wearing_mask_7d features
            continue
        process(f"{source}/{signal}", "state")
        process(f"{source}/{signal}", "county")
    process(f"fb-survey/smoothed_wearing_mask_all", "county")
    process(f"fb-survey/smoothed_wearing_mask_all", "state")


if __name__ == "__main__":
    prepare()
