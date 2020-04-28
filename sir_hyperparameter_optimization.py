import yaml
from cv import cross_validate
import itertools
import pandas as pd
import numpy as np

class Opt:
    config = "cv/nj.yml"
    module = "sir"
    remote = False


WINDOWS = [1, 3, 5, 10, 14, 20, 30]
DISTANCING_REDUCTIONS = np.linspace(0.01, 1, num=10)
RECOVERY_DAYS = [1, 3, 5, 10, 14, 20, 30]


def optimize_hyperparameters():
    cfg = yaml.load(open(Opt.config), Loader=yaml.FullLoader)

    maes = []

    for window, distancing_reduction, recovery_days in (
        itertools.product(WINDOWS, DISTANCING_REDUCTIONS, RECOVERY_DAYS)
    ):
        print("window distancing recovery")
        print(window)
        print(distancing_reduction)
        print(recovery_days)
        cfg["sir"]["train"]["window"] = window
        cfg["sir"]["train"]["distancing_reduction"] = distancing_reduction
        cfg["sir"]["train"]["recovery_days"] = recovery_days

        try:
            df = cross_validate(Opt, cfg=cfg)
        # certain hyperparameters are invalid (lead to NaN)
        except ValueError:
            continue

        mae = df.loc[df["Measure"] == "MAE"].mean(1).values[0]

        maes.append([mae, window, distancing_reduction, recovery_days])

    maes_df = pd.DataFrame(maes, 
        columns=["MAE", "window", "distancing_reduction", "recovery_days"]
    )
    return maes_df


if __name__ == "__main__":
    maes_df = optimize_hyperparameters()
    maes_df.to_csv("sir_hyperparameter_optimization.csv")

