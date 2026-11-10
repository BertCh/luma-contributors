# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference GTWR fits for GPUGeographicallyTemporallyWeightedRegression.

Writes fixtures/gtwr-mgtwr.json next to this script, consumed by
test/gpu-spatial-analysis/spatial-regression/gtwr-mgtwr.spec.ts. Run from the repository root with
    <venv>/bin/python -I modules/analysis/test/oracles/gtwr-mgtwr-oracle.py <mgtwr package dir>
where the argument is the `mgtwr` directory of the mgtwr 2.0.5 source distribution (PyPI sdist;
download with `pip download mgtwr==2.0.5 --no-deps` and unpack, nothing is installed). The venv
needs numpy and scipy. mgtwr imports pandas and joblib at module level for features this script
does not call (data frames, parallel fits), so both are replaced by empty stand-in modules; the
numerical path (GTWRKernel, _compute_betas_gwr, GTWR.fit, GTWRResults) runs unmodified from the
package source. mgtwr declares no licence (PKG-INFO: UNKNOWN): it is executed as a test oracle and
no code of it is copied.
"""

import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import scipy


def load_mgtwr(package_dir):
    for name in ('pandas', 'joblib'):
        module = types.ModuleType(name)
        module.DataFrame = type('DataFrame', (), {})
        module.Series = type('Series', (), {})
        module.Parallel = None
        module.delayed = None
        sys.modules[name] = module
    spec = importlib.util.spec_from_file_location(
        'mgtwr',
        pathlib.Path(package_dir) / '__init__.py',
        submodule_search_locations=[str(package_dir)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules['mgtwr'] = package
    spec.loader.exec_module(package)
    import mgtwr.model

    return package, mgtwr.model


def make_case(seed, rows, columns):
    random = np.random.default_rng(seed)
    coords = random.uniform(0, 10, (rows, 2))
    t = random.uniform(0, 6, (rows, 1))
    x = random.normal(size=(rows, columns))
    beta0 = 2 + 0.2 * coords[:, [0]] + 0.3 * t
    beta1 = 1 + np.sin(coords[:, [1]] / 3) + 0.2 * t
    y = beta0 + beta1 * x[:, [0]] + (0.5 * x[:, [1]] if columns > 1 else 0) + 0.3 * random.normal(size=(rows, 1))
    return coords, t, x, y


def main():
    package, model = load_mgtwr(sys.argv[1])
    cases = []
    for name, seed, rows, columns, bw, tau, kernel in [
        ('gaussian-spacetime', 11, 90, 2, 3.0, 1.5, 'gaussian'),
        ('bisquare-spacetime', 12, 90, 2, 6.0, 2.0, 'bisquare'),
        ('gaussian-spatial-only', 13, 70, 1, 2.5, 0.0, 'gaussian'),
        ('gaussian-time-heavy', 14, 80, 3, 4.0, 8.0, 'gaussian'),
    ]:
        coords, t, x, y = make_case(seed, rows, columns)
        # mgtwr divides by tau in `bw_t`, so tau 0 is passed as 1e-300, which weighs time by 1e-150
        # and leaves the spatial distance unchanged in float64.
        results = model.GTWR(coords, t, x, y, bw, tau if tau > 0 else 1e-300, kernel=kernel, fixed=True).fit()
        cases.append({
            'name': name,
            'rows': rows,
            'columns': columns,
            'kernel': kernel,
            'bandwidth': bw,
            'tau': tau,
            'coords': coords.reshape(-1).tolist(),
            't': t.reshape(-1).tolist(),
            'x': x.reshape(-1).tolist(),
            'y': y.reshape(-1).tolist(),
            'betas': results.betas.reshape(-1).tolist(),
            'fitted': results.predict_value.reshape(-1).tolist(),
            'residuals': results.reside.reshape(-1).tolist(),
            'hat': results.influ.reshape(-1).tolist(),
            'rss': float(results.RSS),
            'trace': float(results.tr_S),
            'r2': float(results.R2),
            'aicc': float(results.aicc),
            'llf': float(results.llf),
            'sigma2': float(results.sigma2),
        })
    multiscale = []
    function = sys.modules['mgtwr.function']
    for name, seed, rows, columns, kernel, initial, bandwidths, taus, tolerance in [
        ('gaussian-three-scales', 21, 80, 2, 'gaussian', (3.0, 1.5), [6.0, 2.5, 4.0], [0.5, 3.0, 1.0], 1e-6),
        ('bisquare-three-scales', 22, 80, 2, 'bisquare', (7.0, 2.0), [9.0, 5.0, 7.0], [1.0, 0.5, 2.0], 1e-6),
    ]:
        coords, t, x, y = make_case(seed, rows, columns)
        design = np.hstack([np.ones((rows, 1)), x])
        size = columns + 1

        def smoother(design_columns, response, bandwidth, tau):
            return model.GTWR(coords, t, design_columns, response, bandwidth, tau, kernel=kernel, fixed=True, constant=False).cal_multi()

        # The component values ride in through the min arguments: sel_func returns (bw_min, tau_min).
        result = function.multi_bws(
            initial[0], initial[1], design, y, rows, size, tolerance, False,
            smoother, lambda *_: None, lambda _selector, bw_min, bw_max, tau_min, tau_max: (bw_min, tau_min),
            bandwidths, bandwidths, taus, taus,
        )
        _bws, _taus, history, _tau_history, scores, betas, err, _bw0, _tau0 = result
        multiscale.append({
            'name': name,
            'rows': rows,
            'columns': columns,
            'kernel': kernel,
            'initial': list(initial),
            'bandwidths': bandwidths,
            'taus': taus,
            'iterations': len(scores),
            'coords': coords.reshape(-1).tolist(),
            't': t.reshape(-1).tolist(),
            'x': x.reshape(-1).tolist(),
            'y': y.reshape(-1).tolist(),
            'betas': betas.reshape(-1).tolist(),
            'residuals': err.reshape(-1).tolist(),
            'scores': [float(value) for value in scores],
        })
    out = pathlib.Path(__file__).parent / 'fixtures' / 'gtwr-mgtwr.json'
    out.write_text(json.dumps({
        'versions': {'mgtwr': package.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__, 'python': sys.version.split()[0]},
        'oracle': 'mgtwr.model.GTWR.fit, fixed bandwidth; weights from mgtwr.kernel.GTWRKernel',
        'cases': cases,
        'multiscale': multiscale,
        'multiscaleOracle': 'mgtwr.function.multi_bws with a fixed (bandwidth, tau) per component',
    }))
    print('wrote', out)


main()
