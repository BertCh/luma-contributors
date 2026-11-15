# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""NumPy float64 eigensolve and analytic Mercator-flow oracle for MOB-06.

Run from the repository root with /usr/bin/python3. No implementation code is imported.
Inputs are rounded to float32 before reference evaluation, as they are on upload.
"""
import json
from pathlib import Path
import numpy as np


def ridge_case(name, values, spacing=(1, 1), minimum_value=0, minimum_curvature=0,
               invalid=(), scale=1, offset=0):
    values = np.asarray(values, dtype=np.float32)
    height, width = values.shape
    input_validity = np.ones_like(values, dtype=np.uint32)
    for row, column in invalid:
        input_validity[row, column] = 0
    calibrated = values.astype(float) * scale + offset
    ridges = np.zeros_like(input_validity)
    validity = np.zeros_like(input_validity)
    strength = np.zeros_like(calibrated)
    spacing_x, spacing_y = spacing
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            stencil = calibrated[row - 1:row + 2, column - 1:column + 2]
            if not np.all(input_validity[row - 1:row + 2, column - 1:column + 2]) or not np.all(np.isfinite(stencil)):
                continue
            gradient = np.array([(stencil[1, 2] - stencil[1, 0]) / (2 * spacing_x),
                                 (stencil[2, 1] - stencil[0, 1]) / (2 * spacing_y)])
            hessian = np.array([[(stencil[1, 2] + stencil[1, 0] - 2 * stencil[1, 1]) / spacing_x**2,
                                 (stencil[2, 2] - stencil[2, 0] - stencil[0, 2] + stencil[0, 0]) / (4 * spacing_x * spacing_y)],
                                [0, (stencil[2, 1] + stencil[0, 1] - 2 * stencil[1, 1]) / spacing_y**2]])
            hessian[1, 0] = hessian[0, 1]
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            eigenvalue = eigenvalues[0]
            normal = eigenvectors[:, 0]
            if eigenvalues[0] == eigenvalues[1]:
                normal = np.array([1., 0.])
            if normal[0] < 0 or (normal[0] == 0 and normal[1] < 0):
                normal = -normal
            validity[row, column] = 1
            strength[row, column] = max(0, -eigenvalue)
            if eigenvalue >= 0 or -eigenvalue < minimum_curvature or stencil[1, 1] < minimum_value:
                continue
            pixel_offset = -(gradient @ normal) / eigenvalue * normal / spacing
            ridges[row, column] = int(np.all(pixel_offset >= -0.5) and np.all(pixel_offset < 0.5))
    return dict(name=name, width=width, height=height, spacing=list(spacing),
                values=[float(value) if np.isfinite(value) else None for value in values.ravel()],
                inputValidity=input_validity.ravel().tolist(), scale=scale, offset=offset,
                minimumValue=minimum_value, minimumCurvature=minimum_curvature,
                ridges=ridges.ravel().tolist(), validity=validity.ravel().tolist(),
                strength=strength.ravel().tolist())


row, column = np.mgrid[-3:4, -4:5]
cases = [
    ridge_case('axis and exact half-cell ties', 30 - (column - .5)**2),
    ridge_case('diagonal anisotropic quadratic', 100 - (2 * column + 3 * row - .7)**2 / 5, spacing=(2, 3)),
    ridge_case('isotropic eigenvalue tie', 30 - column**2 - row**2),
    ridge_case('flat plateau', np.full_like(column, 12)),
    ridge_case('valley', column**2 + 2),
    ridge_case('threshold rejects weak curvature', 30 - column**2, minimum_curvature=3),
    ridge_case('threshold rejects low values', 30 - column**2, minimum_value=31),
    ridge_case('calibrated utilization with missing slot', (30 - column**2) / 2 - 5,
               invalid=[(3, 4)], scale=2, offset=10),
    ridge_case('empty validity slots', 30 - column**2, invalid=list(zip(*np.where(column == column)))),
    ridge_case('utilization gaussian', np.exp(-(column**2 / 2 + row**2 / 30))),
]
nonfinite = (30 - column**2).astype(float)
nonfinite[2, 3] = np.nan
cases.append(ridge_case('nonfinite sample', nonfinite))

# Reviewer regressions: the first Hessian has a tiny negative diagonal eigenvalue beside
# a large positive one. The low-density Hessian has eigenvalues near [-1e-25, 3e-25];
# unscaled float32 discriminant and eigenvector squares underflow. Keep NumPy eigh as the
# independent reference and include a nonzero gradient to exercise the normal direction.
cases.append(ridge_case('small negative eigenvalue survives half-trace cancellation',
                        [[1, 0, 1], [1, 2**-27, 1], [1, 0, 1]]))
small_row, small_column = np.mgrid[-1:2, -1:2]
for amplitude, shift, name in [
    (1e-13, 0, 'low-density Hessian norm avoids underflow'),
    (1e-13, .25, 'low-density nonstationary ridge normal avoids underflow'),
    (1e33, .25, 'large-density nonstationary ridge norm avoids overflow'),
]:
    shifted_column = small_column - shift
    samples = amplitude * (10 + .5 * shifted_column**2 + .5 * small_row**2
                           + 2 * shifted_column * small_row)
    cases.append(ridge_case(name, samples, spacing=(1e6, 1e6)))

# A common Hessian scale must not erase a normal negative diagonal beside a large positive
# one; a nonzero coupling also checks that its small Jacobi correction survives that ratio.
cases.append(ridge_case('wide-range diagonal retains negative curvature',
                        [[0, 0, 0], [5e29, 5e-21, 5e29], [0, 0, 0]]))
cases.append(ridge_case('wide-range coupled Hessian retains negative curvature',
                        [[1e5, 0, -1e5], [5e29, 5e-21, 5e29], [-1e5, 0, 1e5]]))
cases.append(ridge_case('large cell spacing avoids square overflow',
                        [[1e10, 2e10, 1e10]] * 3, spacing=(1e20, 1e20)))

radius = 6378137.

def project(position):
    longitude, latitude = np.deg2rad(np.asarray(position, dtype=float))
    return radius * np.array([longitude, np.arcsinh(np.tan(latitude))])

def unproject(position):
    return np.rad2deg([position[0] / radius, np.arctan(np.sinh(position[1] / radius))])


def flow_case(time_step):
    seeds = np.array([[1, 42], [3.6, 44], [7.7, 47.8], [-1, 42], [1, 86]], dtype=np.float32)
    step_count = 20
    start_time = 0 if time_step > 0 else 200
    positions, classes_start, classes_end, residence, validity = [], [], [], [], []
    no_region = 0xffffffff
    def region(position):
        longitude, latitude = unproject(position)
        return (0 if longitude < 4 else 1) if 0 <= longitude < 8 and 40 <= latitude < 48 else no_region
    for seed in seeds:
        position = project(seed)
        lost = abs(seed[0]) > 180 or abs(seed[1]) > 85.0511287798066 or region(position) == no_region
        start_region = no_region if lost else region(position)
        exited = start_region == no_region
        inside_steps = 0
        for step in range(step_count):
            if lost:
                break
            if not exited:
                if region(position) == start_region:
                    inside_steps += 1
                else:
                    exited = True
            time = start_time + (step + .5) * time_step
            # Uniform, linearly ramping velocity: midpoint RK2 integrates it exactly.
            velocity = np.array([300., 100.]) * (1 + 2 * np.clip(time / 200, 0, 1))
            first_velocity = np.array([300., 100.]) * (1 + 2 * np.clip((start_time + step * time_step) / 200, 0, 1))
            midpoint = position + .5 * time_step * first_velocity
            position += time_step * velocity
            lost = region(midpoint) == no_region or region(position) == no_region
        positions.extend([None, None] if lost else unproject(position).tolist())
        classes_start.append(start_region)
        classes_end.append(no_region if lost else region(position))
        residence.append(inside_steps * abs(time_step))
        validity.append(int(not lost))
    return dict(seeds=seeds.ravel().tolist(), timeStep=time_step, stepCount=step_count,
                startTime=start_time, positions=positions, regionClasses=classes_start + classes_end,
                residence=residence, validity=validity)

output = dict(oracle='NumPy eigh float64 central differences; analytic spherical Mercator and uniform linear-time flow',
              ridgeCases=cases, flowCases=[flow_case(10), flow_case(-10)])
path = Path(__file__).parent / 'fixtures' / 'flow-field-extensions-mob06-numpy.json'
path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
