# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
#
# GSTools 1.7.0 oracle for the metric space-time covariance of GPUSpaceTimeKriging: correlations
# and simple and ordinary kriging predictions and variances. GSTools space-time models are metric
# only, so the other families have no GSTools reference; they are checked against independent
# binary64 solves in the specs. gstat fixtures are pending because R is not available here.
# Environment: CPython 3.14, numpy 2.5.4, gstools 1.7.0 (see the fixture `versions`).
#
# The joint lag is d = sqrt(h^2 + (anisotropy u)^2). GSTools time anisotropy is `anis_t` with a
# temporal length scale len_scale * anis_t, so anisotropy = 1 / anis_t.

import json
from pathlib import Path

import gstools as gs
import numpy as np

rng = np.random.default_rng(20261110)
positions = rng.uniform(-2.0, 2.0, (24, 2))
times = rng.uniform(0.0, 20.0, 24)
values = np.sin(positions[:, 0]) + 0.1 * times + rng.normal(0.0, 0.05, 24)
queries = rng.uniform(-1.5, 1.5, (6, 2))
query_times = rng.uniform(2.0, 18.0, 6)
sill = 1.5
nugget = 0.02
effective_range = 1.8
anisotropy = 0.25
mean = 0.4
factors = {0.5: 3.0, 1.5: 4.749031386012701}


def make_model(nu):
    return gs.Matern(
        dim=3,
        temporal=True,
        var=sill,
        len_scale=np.sqrt(nu) * effective_range / factors[nu],
        nugget=nugget,
        nu=nu,
        anis=[1.0, 1.0 / anisotropy],
    )


cases = {}
for nu in [0.5, 1.5]:
    model = make_model(nu)
    case = {
        "lagPairs": [[0.0, 0.0], [0.5, 0.0], [0.0, 3.0], [0.5, 3.0], [1.2, 6.0]],
        "correlation": [],
    }
    for h, u in case["lagPairs"]:
        d = np.hypot(h, anisotropy * u)
        case["correlation"].append(float(model.cor(np.array([d / model.len_scale]))[0]))
    for mode, unbiased in [("ordinary", True), ("simple", False)]:
        options = {"unbiased": unbiased, "pseudo_inv": False}
        if not unbiased:
            options["mean"] = mean
        krige = gs.krige.Krige(model, np.vstack([positions.T, times]), values, **options)
        prediction, variance = krige(np.vstack([queries.T, query_times]), store=False)
        case[mode] = {"prediction": prediction.tolist(), "variance": variance.tolist()}
    cases[str(nu)] = case

reference = {
    "versions": {"numpy": np.__version__, "gstools": gs.__version__},
    "source": "gstools.Matern with temporal=True and gstools.krige.Krige, CPython 3.14, double precision",
    "positions": positions.tolist(),
    "times": times.tolist(),
    "values": values.tolist(),
    "queries": queries.tolist(),
    "queryTimes": query_times.tolist(),
    "sill": sill,
    "nugget": nugget,
    "range": effective_range,
    "anisotropy": anisotropy,
    "mean": mean,
    "cases": cases,
}
output = Path(__file__).resolve().parent / "fixtures" / "space-time-metric-gstools.json"
output.write_text(json.dumps(reference, indent=2) + "\n")
print(f"wrote {output}")
