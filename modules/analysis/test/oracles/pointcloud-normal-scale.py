# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy eigh + cylinder M3C2 oracle; run with /usr/bin/python3."""
import json
from pathlib import Path
import numpy as np

cases = []
for name, slope, noise in [('flat-ties', 0.0, 0.0), ('tilted-rough-plane', 0.5, 0.02), ('rough-center', 0.0, .04)]:
    normal = np.array([-slope, 0.0, 1.0]) / np.sqrt(1 + slope * slope)
    horizontal = np.array([1.0, 0.0, slope]) / np.sqrt(1 + slope * slope)
    reference = np.array([horizontal * (column * .25) + np.array([0, row * .25, 0]) +
                          normal * (noise * (-1) ** (column + row) *
                                    (1 / (1 + max(abs(column), abs(row))) if name == 'rough-center' else
                                     1 + max(abs(column), abs(row))))
                          for row in range(-2, 3) for column in range(-2, 3)], dtype=np.float32).astype(float)
    comparison = np.asarray(reference + normal * .15, dtype=np.float32).astype(float)
    core_row = 12
    core = reference[core_row]
    radii = [.1, .4, .8]
    candidates = []
    for radius in radii:
        selected = np.flatnonzero(np.sum((reference - core) ** 2, axis=1) <= np.float32(radius ** 2))
        selected = selected[selected != core_row]
        if len(selected) < 3:
            continue
        centered = reference[selected] - reference[selected].mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered / len(selected))
        direction = vectors[:, 0]
        if direction[2] < 0:
            direction = -direction
        candidates.append(dict(radius=radius, variance=max(float(values[0]), 0.0), normal=direction.tolist()))
    chosen = min(candidates, key=lambda candidate: (candidate['variance'], candidate['radius']))
    # Flat eigenvalues are exactly zero, so the smallest eligible radius must win.
    direction = np.array(chosen['normal'])
    projections = []
    for cloud in [reference, comparison]:
        difference = cloud - core
        distances = difference @ direction
        radial = difference - distances[:, None] * direction
        projections.append(distances[(abs(distances) <= .5) & (np.sum(radial ** 2, axis=1) <= 1.1 ** 2)])
    roughness = [float(np.std(distances, ddof=1)) for distances in projections]
    detection = 1.96 * np.sqrt(sum(value ** 2 / len(distances) for value, distances in zip(roughness, projections)))
    difference = float(projections[1].mean() - projections[0].mean())
    cases.append(dict(name=name, reference=reference.tolist(), comparison=comparison.tolist(), coreRow=core_row,
                      radii=radii, candidates=candidates, selected=chosen,
                      counts=[len(distances) for distances in projections],
                      changes=[difference, difference - detection, difference + detection],
                      roughness=roughness + [float(detection)], knownOffset=.15))
Path(__file__).with_name('fixtures').joinpath('pointcloud-normal-scale.json').write_text(json.dumps(
    dict(generator='pointcloud-normal-scale.py', numpy=np.__version__, cases=cases), indent=2) + '\n')

# Independent vectorized least-squares oracle for the finite-grid Gaussian semivariogram fit.
positions = np.array([[0, 0, 0], [.2, .1, 0], [.5, .2, .1], [1, 0, 0], [1.5, .2, 0], [2, .1, .2]])
residuals = np.array([.1, .12, .08, -.1, -.12, -.15])
lengths = [.1, .25, .5, 1, 2]
first, second = np.triu_indices(len(residuals), 1)
squared = np.sum((positions[first] - positions[second]) ** 2, axis=1)
observed = (residuals[first] - residuals[second]) ** 2 / 2
fits = []
for length in lengths:
    model = -np.expm1(-squared / (2 * length ** 2))
    variance = float(np.linalg.lstsq(model[:, None], observed, rcond=None)[0][0])
    fits.append(dict(variance=variance, correlationLength=length,
                     meanSquaredError=float(np.mean((observed - variance * model) ** 2))))
Path(__file__).with_name('fixtures').joinpath('pointcloud-covariance-fit.json').write_text(json.dumps(dict(
    generator='pointcloud-normal-scale.py', numpy=np.__version__, positions=positions.ravel().tolist(),
    residuals=residuals.tolist(), correlationLengths=lengths,
    expected=min(fits, key=lambda fit: (fit['meanSquaredError'], fit['correlationLength']))), indent=2) + '\n')

# PDAL pipeline oracle: horizontal planes with symmetric roughness and a known normal offset.
reference = np.array([[column * .25, row * .25, column * row * .01]
                      for row in range(-2, 3) for column in range(-2, 3)])
comparison = reference + [0, 0, .15]
roughness = [float(np.std(cloud[:, 2])) for cloud in [reference, comparison]]
Path(__file__).with_name('fixtures').joinpath('pointcloud-m3c2-binary.json').write_text(json.dumps(dict(
    generator='pointcloud-normal-scale.py', numpy=np.__version__, reference=reference.tolist(),
    comparison=comparison.tolist(), core=[0, 0, 0], normalRadius=.8, radius=1.1, halfDepth=.5,
    expected=dict(distance=.15, counts=[25, 25], populationRoughness=roughness,
                  uncertainty=float(1.96 * np.sqrt(sum(value ** 2 / 25 for value in roughness))), significant=1)), indent=2) + '\n')
