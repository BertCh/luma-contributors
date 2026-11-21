# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/audit-regression-helpers.oracle.py
Independent SciPy distribution-function literals. Matrix fixtures use audit-ols.oracle.py.
"""
import json
from scipy import special, stats

if __name__ == '__main__':
    print(json.dumps(dict(
        gamma=[dict(freedom=freedom,value=special.gammaln(freedom/2)) for freedom in [1,2,3,4,15,30,101]],
        chiSquare=[dict(value=value,freedom=freedom,survival=stats.chi2.sf(value,freedom))
                   for value,freedom in [(0.1,1),(2,3),(10,15),(30,15),(100,15),(50,30)]],
        student=[dict(statistic=value,freedom=freedom,survival=stats.t.sf(value,freedom),
                      twoSided=2*stats.t.sf(abs(value),freedom))
                 for value,freedom in [(-4.5,6),(-0.1,1),(0,6),(2.5,6),(8,15),(40,101)]]
    ),indent=2))
