# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Exact rational geometry for the one-row cumulative-visibility status fixture.

Run: /usr/bin/python3 modules/analysis/test/gpu-terrain/terrain-analysis/terrain-analysis-status-oracles.oracle.py

The documented sight-line samples are integer columns between observer and target. For each one,
extend the vertical gap between terrain and the observer-target line to the target
plane by similar triangles. The minimum extended gap is target-plane clearance.
Clearance >= tolerance is visible, clearance < -tolerance is hidden, and the band
between those boundaries is marginal. This uses exact rational geometry rather
than float32 shader code, a CPU twin, or values captured from a GPU run.
"""

from fractions import Fraction
import json


def derive_counts(wall, tolerance, observers=(0, 0)):
    elevation = [Fraction(0) for _ in range(12)]
    elevation[6] = Fraction(wall)
    visible = [0] * 12
    marginal = [0] * 12
    clearances = []
    for target in range(12):
        target_clearances = []
        for observer in observers:
            eye = elevation[observer] + 1
            distance = abs(target - observer)
            direction = 1 if target > observer else -1
            gaps = []
            for step in range(1, distance):
                sample = observer + direction * step
                line_height = eye + (elevation[target] - eye) * Fraction(step, distance)
                gaps.append((line_height - elevation[sample]) * Fraction(distance, step))
            clearance = min(gaps) if gaps else None
            is_visible = clearance is None or clearance >= tolerance
            is_marginal = not is_visible and clearance >= -tolerance
            visible[target] += int(is_visible)
            marginal[target] += int(is_marginal)
            target_clearances.append(str(clearance) if clearance is not None else "no interior sample")
        clearances.append(target_clearances)
    return {"visible": visible, "marginal": marginal, "clearances": clearances}


if __name__ == "__main__":
    for name, wall, tolerance, observers in [
        ("wall with tolerance", Fraction(1, 2), Fraction(1, 2), (0, 0)),
        ("flat with tolerance", 0, Fraction(1, 2), (0, 0)),
        ("flat without tolerance", 0, Fraction(0), (0, 0)),
        ("opposite observers with hidden and marginal rays", 2, Fraction(1, 2), (0, 10)),
    ]:
        result = derive_counts(wall, tolerance, observers)
        if observers == (0, 10):
            assert (result["visible"][10], result["marginal"][10]) == (1, 0)
            assert (result["visible"][4], result["marginal"][4]) == (0, 1)
        print(json.dumps({"name": name, **result}))
