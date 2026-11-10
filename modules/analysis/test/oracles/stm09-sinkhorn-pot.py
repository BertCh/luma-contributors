# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""POT reference for GPUSinkhornDivergence (STM-09).

Writes fixtures/stm09-sinkhorn-pot.json next to this script, consumed by
test/gpu-spatial-analysis/optimal-transport/stm09-sinkhorn-divergence.spec.ts. Run from the
repository root with
    python -I modules/analysis/test/oracles/stm09-sinkhorn-pot.py
in an environment with numpy, scipy and POT==0.9.7.post1 (MIT; pinned in requirements.txt). If POT
is unpacked into a directory of its own instead of installed, set STM09_POT_SITE to that directory.

Definition (Genevay et al. 2018; Feydy et al. 2019), for masses a, b on a grid with squared
Euclidean cost C between cell centers:
    OT(a, b) = min_P <P, C> + epsilon KL(P | a x b),  P with marginals a and b,
    S(a, b)  = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2.
The coupling P comes from ot.sinkhorn(method='sinkhorn_log') at a 1e-13 marginal tolerance, and OT
is evaluated from P by the formula above, so the quantity is POT's coupling and this script's
definition of the cost. The support of a and b is passed to POT (zero cells carry no coupling).

POT's own ot.bregman.empirical_sinkhorn_divergence is NOT this quantity: its documentation says it
"does not account for the entropic contributions" and returns <P, C> - (<P_a, C> + <P_b, C>) / 2.
It is recorded as `potPrimalOnlyDivergence` for contrast only.
"""

import json
import os
import platform
import sys
import warnings

import numpy as np

if os.environ.get('STM09_POT_SITE'):
    sys.path.insert(0, os.environ['STM09_POT_SITE'])
warnings.filterwarnings('ignore', category=SyntaxWarning)
import ot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'stm09-sinkhorn-pot.json')


def f32(values):
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def blob(width, height, center_x, center_y, deviation, floor=0.0):
    ys = np.arange(width * height) // width
    xs = np.arange(width * height) % width
    values = np.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2 * deviation**2))
    values[values < floor] = 0.0
    return values


def cell_centers(width, height, cell_width, cell_height):
    indices = np.arange(width * height)
    return np.stack([(indices % width) * cell_width, (indices // width) * cell_height], axis=1)


def entropic_cost(a, b, centers, epsilon):
    """OT(a, b) = <P, C> + epsilon KL(P | a x b) from POT's coupling, on the support of a and b."""
    a = a / a.sum()
    b = b / b.sum()
    rows = np.flatnonzero(a > 0)
    columns = np.flatnonzero(b > 0)
    cost = ot.dist(centers[rows], centers[columns], metric='sqeuclidean')
    coupling = ot.sinkhorn(
        a[rows], b[columns], cost, epsilon, method='sinkhorn_log', numItermax=1000000, stopThr=1e-13
    )
    transport = float(np.sum(coupling * cost))
    positive = coupling > 0
    reference = np.outer(a[rows], b[columns])
    divergence = float(np.sum(coupling[positive] * np.log(coupling[positive] / reference[positive])))
    marginal_error = float(
        np.abs(coupling.sum(axis=1) - a[rows]).sum() + np.abs(coupling.sum(axis=0) - b[columns]).sum()
    )
    assert marginal_error < 1e-9, marginal_error
    return transport + epsilon * divergence, marginal_error


def build_case(name, width, height, cell_width, cell_height, epsilon, source, target):
    source = f32(source)
    target = f32(target)
    centers = cell_centers(width, height, cell_width, cell_height)
    cross, cross_error = entropic_cost(source, target, centers, epsilon)
    self_source, source_error = entropic_cost(source, source, centers, epsilon)
    self_target, target_error = entropic_cost(target, target, centers, epsilon)
    rows = np.flatnonzero(source > 0)
    columns = np.flatnonzero(target > 0)
    primal_only = float(
        ot.bregman.empirical_sinkhorn_divergence(
            centers[rows],
            centers[columns],
            epsilon,
            a=source[rows] / source.sum(),
            b=target[columns] / target.sum(),
            numIterMax=1000000,
            stopThr=1e-13,
        )
    )
    max_cost = float(((width - 1) * cell_width) ** 2 + ((height - 1) * cell_height) ** 2)
    return {
        'name': name,
        'width': width,
        'height': height,
        'cellWidth': cell_width,
        'cellHeight': cell_height,
        'epsilon': epsilon,
        'maximumCostOverEpsilon': max_cost / epsilon,
        'source': source.tolist(),
        'target': target.tolist(),
        'crossCost': cross,
        'sourceCost': self_source,
        'targetCost': self_target,
        'divergence': cross - self_source / 2 - self_target / 2,
        'potPrimalOnlyDivergence': primal_only,
        'potMarginalError': max(cross_error, source_error, target_error),
    }


def main():
    cases = []
    cases.append(build_case(
        'blobs-8x8-epsilon-2', 8, 8, 1.0, 1.0, 2.0,
        blob(8, 8, 2, 2, 1.2, 1e-3), blob(8, 8, 5, 4, 1.5, 1e-3),
    ))
    cases.append(build_case(
        'blobs-8x8-epsilon-0.5', 8, 8, 1.0, 1.0, 0.5,
        blob(8, 8, 2, 2, 1.2), blob(8, 8, 5, 4, 1.5),
    ))
    cases.append(build_case(
        'anisotropic-cells-12x9-epsilon-1', 12, 9, 1.0, 0.5, 1.0,
        blob(12, 9, 3, 2, 1.5, 1e-3), blob(12, 9, 8, 6, 2.0, 1e-3),
    ))
    squares_source = np.zeros(100)
    squares_target = np.zeros(100)
    for y in range(1, 4):
        for x in range(1, 4):
            squares_source[y * 10 + x] = 1.0 + 0.1 * (x + y)
    for y in range(6, 9):
        for x in range(6, 9):
            squares_target[y * 10 + x] = 2.0 - 0.1 * (x - y)
    cases.append(build_case('disjoint-squares-10x10-epsilon-4', 10, 10, 1.0, 1.0, 4.0, squares_source, squares_target))
    identical = blob(8, 8, 3, 3, 1.5, 1e-3)
    cases.append(build_case('identical-8x8-epsilon-2', 8, 8, 1.0, 1.0, 2.0, identical, identical))
    delta_source = np.zeros(64)
    delta_target = np.zeros(64)
    delta_source[1 * 8 + 1] = 3.0
    delta_target[4 * 8 + 5] = 0.5
    cases.append(build_case('deltas-8x8-epsilon-2', 8, 8, 1.0, 1.0, 2.0, delta_source, delta_target))
    bimodal = blob(16, 16, 4, 4, 1.8, 1e-4) + 0.6 * blob(16, 16, 11, 5, 1.8, 1e-4)
    cases.append(build_case(
        'bimodal-16x16-epsilon-6', 16, 16, 1.0, 1.0, 6.0, bimodal, blob(16, 16, 8, 10, 3.0),
    ))
    # Closed forms checked here, independently of POT: two deltas at squared distance d2 have
    # OT(a, b) = d2 (the coupling is one cell, KL = 0) and OT(a, a) = 0, so S = d2 for any epsilon.
    deltas = cases[5]
    assert abs(deltas['divergence'] - 25.0) < 1e-9, deltas['divergence']
    assert abs(cases[4]['divergence']) < 1e-9, cases[4]['divergence']
    for case in cases:
        assert case['divergence'] >= -1e-12, case['name']

    fixture = {
        'generator': 'test/oracles/stm09-sinkhorn-pot.py',
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'POT': ot.__version__,
        },
        'definition': 'S = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2 with OT = <P, C> + epsilon KL(P | a x b), squared Euclidean cost between cell centers, P from ot.sinkhorn(method="sinkhorn_log", stopThr=1e-13)',
        'cases': cases,
    }
    with open(OUTPUT, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')
    print(f'wrote {OUTPUT}: {len(cases)} cases')
    for case in cases:
        print(f"  {case['name']}: S = {case['divergence']:.9g}, POT primal-only {case['potPrimalOnlyDivergence']:.9g}")


if __name__ == '__main__':
    main()
