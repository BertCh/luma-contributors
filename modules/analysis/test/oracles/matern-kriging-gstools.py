# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
#
# GSTools 1.7.0 oracle for the Matern covariance of GPULocalKriging: correlation values (the
# library's Bessel-function form) and simple and ordinary kriging predictions and variances.
# Environment: CPython 3.14, numpy 2.5.4, gstools 1.7.0 (see the fixture `versions`).
#
# GSTools parameterizes Matern by len_scale with the Bessel argument sqrt(nu) * h / len_scale. The
# contributor's effective range gives the closed-form argument a = factor * h / range, so
# len_scale = sqrt(nu) * range / factor, with factor 3 at nu = 1/2 and the double-precision roots
# of a - ln(1 + a) = 3 at nu = 3/2 and a - ln(1 + a + a^2 / 3) = 3 at nu = 5/2.

import json
from pathlib import Path

import gstools as gs
import numpy as np

rng = np.random.default_rng(20261109)
positions = rng.uniform(-2.0, 2.0, (28, 2))
values = np.sin(1.3 * positions[:, 0]) + 0.6 * np.cos(positions[:, 1]) + rng.normal(0.0, 0.05, 28)
queries = rng.uniform(-1.8, 1.8, (9, 2))
sill = 2.0
nugget = 0.05
effective_range = 1.6
mean = 0.5
distances = np.array([1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.6, 2.5, 4.0])

MATERN_3_2_RANGE_FACTOR = 4.749031386012701
MATERN_5_2_RANGE_FACTOR = 5.924462618951797
factors = {0.5: 3.0, 1.5: MATERN_3_2_RANGE_FACTOR, 2.5: MATERN_5_2_RANGE_FACTOR}


def make_model(nu):
    return gs.Matern(
        dim=2,
        var=sill,
        len_scale=np.sqrt(nu) * effective_range / factors[nu],
        nugget=nugget,
        nu=nu,
    )


cases = {}
for nu in [0.5, 1.5, 2.5]:
    model = make_model(nu)
    correlation = model.cor(distances / model.len_scale).tolist()
    case = {"lenScale": float(model.len_scale), "correlation": correlation}
    for mode, unbiased in [("ordinary", True), ("simple", False)]:
        options = {"unbiased": unbiased, "pseudo_inv": False}
        if not unbiased:
            options["mean"] = mean
        krige = gs.krige.Krige(model, positions.T, values, **options)
        prediction, variance = krige(queries.T, store=False)
        case[mode] = {
            "prediction": prediction.tolist(),
            "variance": variance.tolist(),
        }
    cases[str(nu)] = case

reference = {
    "versions": {"numpy": np.__version__, "gstools": gs.__version__},
    "source": "gstools.Matern and gstools.krige.Krige, CPython 3.14, double precision",
    "positions": positions.tolist(),
    "values": values.tolist(),
    "queries": queries.tolist(),
    "sill": sill,
    "nugget": nugget,
    "range": effective_range,
    "mean": mean,
    "distances": distances.tolist(),
    "cases": cases,
}
output = Path(__file__).resolve().parent / "fixtures" / "matern-kriging-gstools.json"
output.write_text(json.dumps(reference, indent=2) + "\n")
print(f"wrote {output}")
