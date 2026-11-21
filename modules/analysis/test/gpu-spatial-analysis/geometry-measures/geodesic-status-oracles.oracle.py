# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent first-step Vincenty witnesses, with binary64 and float32 equations.

Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-measures/geodesic-status-oracles.oracle.py
The spec pins status literals, never imports implementation or executes Python.
"""
import math
import numpy as np

flattening = 1 / 298.257223563
major = 6378137.0
minor = major * (1 - flattening)
# Equatorial inverse: reduced latitudes and cos^2(alpha) vanish. The first
# update is lambda_1 = L + f*L, from initial lambda_0 = L.
inverseChange = flattening * math.pi / 180
print('inverse first change', inverseChange)
assert inverseChange > 1e-6
# Direct, origin 45 degrees, bearing north, distance 1000 km.
reduced = math.atan((1 - flattening) * math.tan(math.pi / 4))
sigmaOne = reduced
square = (major * major - minor * minor) / (minor * minor)
coefficientA = 1 + square / 16384 * (4096 + square * (-768 + square * (320 - 175 * square)))
coefficientB = square / 1024 * (256 + square * (-128 + square * (74 - 47 * square)))
sigma = 1000000 / (minor * coefficientA)
cosineMiddle = math.cos(2 * sigmaOne + sigma)
delta = coefficientB * math.sin(sigma) * (cosineMiddle + coefficientB / 4 * (
    math.cos(sigma) * (-1 + 2 * cosineMiddle ** 2) - coefficientB / 6 * cosineMiddle *
    (-3 + 4 * math.sin(sigma) ** 2) * (-3 + 4 * cosineMiddle ** 2)))
print('direct first change', abs(delta))
assert abs(delta) > 1e-6
# An absolute 1e-6 stopping rule falsely accepts these shorter trips. The
# declared relative/scale-aware thresholds still require another iteration.
longitude = math.radians(0.01)
shortInverseChange = flattening * longitude
inverseThreshold = 4.8e-7 * abs(longitude + shortInverseChange)
assert inverseThreshold < shortInverseChange < 1e-6
print('short inverse change / relative threshold', shortInverseChange, inverseThreshold)
# At the equator bearing north, sigma_1=0 and cos^2(alpha)=1. Coefficients A/B
# above depend only on cos^2(alpha), which was also 1 for the 45N northward trip.
sigma = 2000 / (minor * coefficientA)
cosineMiddle = math.cos(sigma)
shortDirectChange = abs(coefficientB * math.sin(sigma) * (
    cosineMiddle + coefficientB / 4 * (
        math.cos(sigma) * (-1 + 2 * cosineMiddle ** 2)
        - coefficientB / 6 * cosineMiddle
        * (-3 + 4 * math.sin(sigma) ** 2) * (-3 + 4 * cosineMiddle ** 2)
    )
))
directThreshold = 2.4e-7 * max(1, sigma + shortDirectChange)
assert directThreshold < shortDirectChange < 1e-6
print('short direct change / scale-aware threshold', shortDirectChange, directThreshold)
# Removing the inverse scale factor but keeping its coefficient 4.8e-7
# still rejects the 0.01-degree row. A tenfold shorter inverse separates them.
longitude = math.radians(0.001)
tinyInverseChange = flattening * longitude
tinyInverseThreshold = 4.8e-7 * abs(longitude + tinyInverseChange)
assert tinyInverseThreshold < tinyInverseChange < 4.8e-7
print('tiny inverse change / relative threshold', tinyInverseChange, tinyInverseThreshold)

# Independent first Vincenty direct update, equatorial origin with bearing
# 85.75 degrees. Unlike the northward examples, cos^2(alpha) is now small.
# Round each scalar expression to float32 to expose the two quantized sigma
# values; the binary64 equations also accept this step under scaled stopping.
scalar = np.float32
bearing = scalar(scalar(85.75) * scalar(math.pi / 180))
sineAlpha = scalar(np.sin(bearing))
cosineSquaredAlpha = scalar(scalar(1) - scalar(sineAlpha * sineAlpha))
square = scalar(cosineSquaredAlpha * scalar((major * major - minor * minor) / (minor * minor)))
coefficientA = scalar(scalar(1) + scalar(square / scalar(16384)) * scalar(
    scalar(4096) + square * scalar(scalar(-768) + square * scalar(scalar(320) - scalar(175) * square))
))
coefficientB = scalar(square / scalar(1024) * scalar(
    scalar(256) + square * scalar(scalar(-128) + square * scalar(scalar(74) - scalar(47) * square))
))
sigma = scalar(scalar(19500000) / scalar(scalar(minor) * coefficientA))
sineSigma, cosineSigma = scalar(np.sin(sigma)), scalar(np.cos(sigma))
# sigma_1 is exactly zero at the equator, so cos(2*sigma_1+sigma)=cos(sigma).
delta = scalar(coefficientB * sineSigma * scalar(
    cosineSigma + scalar(coefficientB / scalar(4)) * scalar(
        cosineSigma * scalar(scalar(-1) + scalar(2) * cosineSigma * cosineSigma)
        - scalar(coefficientB / scalar(6)) * cosineSigma
        * scalar(scalar(-3) + scalar(4) * sineSigma * sineSigma)
        * scalar(scalar(-3) + scalar(4) * cosineSigma * cosineSigma)
    )
))
updatedSigma = scalar(sigma + delta)
longDirectChange = abs(scalar(updatedSigma - sigma))
longDirectThreshold = scalar(scalar(2.4e-7) * max(scalar(1), updatedSigma))
assert sigma > 1
assert scalar(2.4e-7) < longDirectChange <= longDirectThreshold
assert longDirectChange > shortDirectChange
print('long direct sigma / updated sigma', sigma, updatedSigma)
print('long direct change / scaled threshold', longDirectChange, longDirectThreshold)
# Reject a smaller change (shortDirectChange) yet accept this larger one:
# no fixed direct tolerance can implement both stopping decisions.
# Coincident inverse and zero-distance direct have exactly zero change.
# Equatorial antipodes are singular before iteration, not budget exhaustion.
