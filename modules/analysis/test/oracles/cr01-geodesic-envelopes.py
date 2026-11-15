# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Regenerate CR01 PROJ oracles and a nominal IEEE float32 measurement.

Run with /usr/bin/python3. This measures the arithmetic model, not a GPU driver.
The headless spec is the required hardware envelope check (transcendentals/fast math).
"""
import json
from pathlib import Path

import numpy
import pyproj

FLOAT = numpy.float32
DEGREES = FLOAT(numpy.pi / 180)
RADIANS = FLOAT(180 / numpy.pi)
FLATTENING = FLOAT(1 / 298.257223563)
SEMI_MINOR = FLOAT(6378137 * (1 - 1 / 298.257223563))
ECCENTRICITY = FLOAT((6378137 ** 2 - (6378137 * (1 - 1 / 298.257223563)) ** 2) / (6378137 * (1 - 1 / 298.257223563)) ** 2)
RADIUS = 6371008.8


def longitude_delta(first, second):
    result = (second - FLOAT(180) * numpy.rint((second - first) / FLOAT(360))) - (first + FLOAT(180) * numpy.rint((second - first) / FLOAT(360)))
    seam = (numpy.abs(first) > 90) & (numpy.abs(first) <= 180) & (numpy.abs(second) > 90) & (numpy.abs(second) <= 180) & ((first < 0) != (second < 0))
    # Integer seam branch from the shared kernel; conversion is exact for f32 endpoints.
    units = (second * FLOAT(131072)).astype(numpy.int64) - (first * FLOAT(131072)).astype(numpy.int64) + numpy.where(first > 0, 47185920, -47185920)
    return numpy.where(seam, units.astype(FLOAT) * FLOAT(1 / 131072), result).astype(FLOAT)


def cosine_latitude(latitude):
    return numpy.sin((FLOAT(90) - numpy.abs(latitude)) * DEGREES)


def central_angle(first, second):
    first_latitude = first[:, 1] * DEGREES
    second_latitude = second[:, 1] * DEGREES
    latitude_delta = (second[:, 1] - first[:, 1]) * DEGREES
    longitude = longitude_delta(first[:, 0], second[:, 0]) * DEGREES
    half_versine = FLOAT(2) * numpy.sin(FLOAT(.5) * longitude) ** 2
    east = cosine_latitude(second[:, 1]) * numpy.sin(longitude)
    north = numpy.sin(latitude_delta) + numpy.sin(first_latitude) * cosine_latitude(second[:, 1]) * half_versine
    vertical = numpy.sin(first_latitude) * numpy.sin(second_latitude) + cosine_latitude(first[:, 1]) * cosine_latitude(second[:, 1]) * (FLOAT(1) - half_versine)
    degrees_delta = longitude_delta(first[:, 0], second[:, 0])
    supplement = (FLOAT(180) - numpy.abs(degrees_delta)) * DEGREES
    half_sine = numpy.sin(FLOAT(.5) * supplement)
    opposite = (numpy.abs(degrees_delta) > 90) & (first[:, 1] * second[:, 1] <= 0)
    east = numpy.where(opposite, cosine_latitude(second[:, 1]) * numpy.sign(degrees_delta) * numpy.sin(supplement), east)
    north = numpy.where(opposite, numpy.sin((first[:, 1] + second[:, 1]) * DEGREES) - FLOAT(2) * numpy.sin(first_latitude) * cosine_latitude(second[:, 1]) * half_sine ** 2, north)
    vertical = numpy.where(opposite, numpy.sin(first_latitude) * numpy.sin(second_latitude) + cosine_latitude(first[:, 1]) * cosine_latitude(second[:, 1]) * (FLOAT(2) * half_sine ** 2 - FLOAT(1)), vertical)
    coincident = (first[:, 1] == second[:, 1]) & ((degrees_delta == 0) | (numpy.abs(first[:, 1]) == 90))
    return numpy.where(coincident, FLOAT(0), numpy.arctan2(numpy.hypot(east, north), vertical))


def reduced_cosine(latitude_degrees, latitude_radians):
    absolute_latitude = numpy.abs(latitude_radians)
    sine = numpy.sin(absolute_latitude)
    cosine = numpy.cos(absolute_latitude)
    gap = numpy.arctan(FLATTENING * sine * cosine / (cosine * cosine + (FLOAT(1) - FLATTENING) * sine * sine))
    colatitude = (FLOAT(90) - numpy.abs(latitude_degrees)) * DEGREES
    ordinary = numpy.sin(colatitude + gap)
    normalized = numpy.sin(colatitude) / numpy.hypot(numpy.sin(colatitude), (FLOAT(1) - FLATTENING) * numpy.sin(latitude_radians))
    return numpy.where(numpy.abs(latitude_degrees) > FLOAT(89.999), normalized, ordinary)


def measure_inverse(first, second):
    first_latitude = first[:, 1] * DEGREES
    second_latitude = second[:, 1] * DEGREES

    def gap(latitude):
        sine = numpy.sin(latitude)
        cosine = numpy.cos(latitude)
        return numpy.arctan(FLATTENING * sine * cosine / (cosine * cosine + (FLOAT(1) - FLATTENING) * sine * sine))

    first_gap = gap(first_latitude)
    second_gap = gap(second_latitude)
    first_sine = numpy.sin(first_latitude - first_gap)
    second_sine = numpy.sin(second_latitude - second_gap)
    first_cosine = reduced_cosine(first[:, 1], first_latitude)
    second_cosine = reduced_cosine(second[:, 1], second_latitude)
    latitude_delta = (second[:, 1] - first[:, 1]) * DEGREES
    reduced_delta = latitude_delta - (second_gap - first_gap)
    mean_sine = numpy.sin(FLOAT(.5) * (first_latitude + second_latitude))
    mean_cosine = numpy.cos(FLOAT(.5) * (first_latitude + second_latitude))
    reduced_delta = numpy.where(numpy.abs(latitude_delta) < FLOAT(.01), latitude_delta * (FLOAT(1) - FLATTENING) / (mean_cosine ** 2 + (FLOAT(1) - FLATTENING) ** 2 * mean_sine ** 2), reduced_delta)
    longitude = longitude_delta(first[:, 0], second[:, 0]) * DEGREES
    auxiliary = longitude.copy()
    converged = numpy.zeros(len(first), dtype=bool)
    for iteration in range(17):
        sine_longitude = numpy.sin(auxiliary)
        half_versine = FLOAT(2) * numpy.sin(FLOAT(.5) * auxiliary) ** 2
        north = numpy.sin(reduced_delta) + first_sine * second_cosine * half_versine
        east = second_cosine * sine_longitude
        sine_sigma = numpy.hypot(east, north)
        cosine_sigma = first_sine * second_sine + first_cosine * second_cosine * (FLOAT(1) - half_versine)
        sigma = numpy.arctan2(sine_sigma, cosine_sigma)
        with numpy.errstate(invalid='ignore', divide='ignore'):
            sine_alpha = first_cosine * second_cosine * sine_longitude / sine_sigma
            squared_alpha = FLOAT(1) - sine_alpha ** 2
            cosine_middle = numpy.where(squared_alpha != 0, cosine_sigma - FLOAT(2) * first_sine * second_sine / squared_alpha, FLOAT(0))
        if iteration == 16:
            break
        correction = FLATTENING / FLOAT(16) * squared_alpha * (FLOAT(4) + FLATTENING * (FLOAT(4) - FLOAT(3) * squared_alpha))
        updated = longitude + (FLOAT(1) - correction) * FLATTENING * sine_alpha * (sigma + correction * sine_sigma * (cosine_middle + correction * cosine_sigma * (FLOAT(-1) + FLOAT(2) * cosine_middle ** 2)))
        updated = numpy.where(converged, auxiliary, updated)
        converged |= numpy.abs(updated - auxiliary) <= FLOAT(4.8e-7) * numpy.abs(updated)
        auxiliary = updated
    squared_parameter = squared_alpha * ECCENTRICITY
    first_coefficient = FLOAT(1) + squared_parameter / FLOAT(16384) * (FLOAT(4096) + squared_parameter * (FLOAT(-768) + squared_parameter * (FLOAT(320) - FLOAT(175) * squared_parameter)))
    second_coefficient = squared_parameter / FLOAT(1024) * (FLOAT(256) + squared_parameter * (FLOAT(-128) + squared_parameter * (FLOAT(74) - FLOAT(47) * squared_parameter)))
    sigma_delta = second_coefficient * sine_sigma * (cosine_middle + second_coefficient / FLOAT(4) * (cosine_sigma * (FLOAT(-1) + FLOAT(2) * cosine_middle ** 2) - second_coefficient / FLOAT(6) * cosine_middle * (FLOAT(-3) + FLOAT(4) * sine_sigma ** 2) * (FLOAT(-3) + FLOAT(4) * cosine_middle ** 2)))
    distance = SEMI_MINOR * first_coefficient * (sigma - sigma_delta)
    initial = numpy.arctan2(second_cosine * sine_longitude, north) * RADIANS
    final = numpy.arctan2(first_cosine * sine_longitude, numpy.sin(reduced_delta) - first_cosine * second_sine * half_versine) * RADIANS
    coincident = (first[:, 1] == second[:, 1]) & ((longitude == 0) | (numpy.abs(first[:, 1]) == 90))
    distance[coincident] = 0
    initial[coincident] = 0
    final[coincident] = 0
    converged[coincident] = True
    converged &= FLOAT(numpy.pi) - central_angle(first, second) >= FLOAT(.02)
    return distance, initial, final, converged


def measure_sphere(first, second):
    longitude = longitude_delta(first[:, 0], second[:, 0]) * DEGREES
    latitude_delta = (second[:, 1] - first[:, 1]) * DEGREES
    half_sine = numpy.sin(longitude * FLOAT(.5))
    east = numpy.sin(longitude) * cosine_latitude(second[:, 1])
    north = numpy.sin(latitude_delta) + FLOAT(2) * numpy.sin(first[:, 1] * DEGREES) * cosine_latitude(second[:, 1]) * half_sine ** 2
    degrees_delta = longitude_delta(first[:, 0], second[:, 0])
    supplement = (FLOAT(180) - numpy.abs(degrees_delta)) * DEGREES
    opposite = (numpy.abs(degrees_delta) > 90) & (first[:, 1] * second[:, 1] <= 0)
    east = numpy.where(opposite, cosine_latitude(second[:, 1]) * numpy.sign(degrees_delta) * numpy.sin(supplement), east)
    north = numpy.where(opposite, numpy.sin((first[:, 1] + second[:, 1]) * DEGREES) - FLOAT(2) * numpy.sin(first[:, 1] * DEGREES) * cosine_latitude(second[:, 1]) * numpy.sin(FLOAT(.5) * supplement) ** 2, north)
    initial = numpy.arctan2(east, north) * RADIANS
    angle = central_angle(first, second)
    vectors = []
    for positions in [first, second]:
        cosine = cosine_latitude(positions[:, 1])
        vectors.append(numpy.column_stack([cosine * numpy.cos(positions[:, 0] * DEGREES), cosine * numpy.sin(positions[:, 0] * DEGREES), numpy.sin(positions[:, 1] * DEGREES)]))
    with numpy.errstate(invalid='ignore', divide='ignore'):
        weight = numpy.sin(angle * FLOAT(.5)) / numpy.sin(angle)
    weight = numpy.where(angle < FLOAT(1e-6), FLOAT(.5), weight)
    vector = vectors[0] * weight[:, None] + vectors[1] * weight[:, None]
    midpoint = numpy.column_stack([numpy.arctan2(vector[:, 1], vector[:, 0]) * RADIANS, numpy.arctan2(vector[:, 2], numpy.hypot(vector[:, 0], vector[:, 1])) * RADIANS])
    return angle * FLOAT(RADIUS), initial, midpoint


def measure_direct(origins, bearings, distances, ellipsoidal, local_offsets=False):
    latitude = origins[:, 1] * DEGREES
    azimuth = bearings * DEGREES
    sine_azimuth = numpy.sin(azimuth)
    cosine_azimuth = numpy.cos(azimuth)
    cosine = cosine_latitude(origins[:, 1])
    sine = numpy.sin(latitude)
    if not ellipsoidal:
        angle = distances / FLOAT(RADIUS)
        east = sine_azimuth * numpy.sin(angle)
        north = cosine * numpy.cos(angle) - sine * numpy.sin(angle) * cosine_azimuth
        vertical = sine * numpy.cos(angle) + cosine * numpy.sin(angle) * cosine_azimuth
        result = numpy.column_stack([origins[:, 0] + numpy.arctan2(east, north) * RADIANS, numpy.arctan2(vertical, numpy.hypot(east, north)) * RADIANS])
        half_sine = numpy.sin(FLOAT(.5) * angle)
        sine_delta = FLOAT(-2) * sine * half_sine * half_sine + cosine * numpy.sin(angle) * cosine_azimuth
        destination_sine = numpy.clip(sine + sine_delta, FLOAT(-1), FLOAT(1))
        rough_latitude = numpy.arcsin(destination_sine)
        midpoint_cosine = numpy.cos(FLOAT(.5) * (latitude + rough_latitude))
        with numpy.errstate(invalid='ignore', divide='ignore'):
            latitude_change = numpy.where(numpy.abs(midpoint_cosine) > FLOAT(1e-3), FLOAT(2) * numpy.arcsin(numpy.clip(sine_delta / (FLOAT(2) * midpoint_cosine), FLOAT(-1), FLOAT(1))), rough_latitude - latitude)
        longitude_change = numpy.arctan2(sine_azimuth * numpy.sin(angle) * cosine, numpy.cos(angle) - sine * destination_sine)
        ordinary = numpy.column_stack([origins[:, 0] + longitude_change * RADIANS, origins[:, 1] + latitude_change * RADIANS])
        polar = (numpy.abs(angle) >= FLOAT(numpy.pi / 4)) | (numpy.abs(origins[:, 1]) >= FLOAT(80)) | (numpy.abs(destination_sine) >= FLOAT(numpy.sin(numpy.deg2rad(80))))
        result = numpy.where(polar[:, None], result, ordinary)
        result[distances == 0] = origins[distances == 0]
        arrival_north = sine * numpy.sin(angle) - cosine * numpy.cos(angle) * cosine_azimuth
        arrival = numpy.arctan2(cosine * sine_azimuth, -arrival_north) * RADIANS
        arrival[distances == 0] = FLOAT(180)
        return result, arrival
    reduced = latitude - numpy.arctan(FLATTENING * sine * numpy.cos(latitude) / (numpy.cos(latitude) ** 2 + (FLOAT(1) - FLATTENING) * sine ** 2))
    sine = numpy.sin(reduced)
    cosine = reduced_cosine(origins[:, 1], latitude)
    first_sigma = numpy.arctan2(sine, cosine * cosine_azimuth)
    sine_alpha = cosine * sine_azimuth
    squared_alpha = FLOAT(1) - sine_alpha ** 2
    squared_parameter = squared_alpha * ECCENTRICITY
    first_coefficient = FLOAT(1) + squared_parameter / FLOAT(16384) * (FLOAT(4096) + squared_parameter * (FLOAT(-768) + squared_parameter * (FLOAT(320) - FLOAT(175) * squared_parameter)))
    second_coefficient = squared_parameter / FLOAT(1024) * (FLOAT(256) + squared_parameter * (FLOAT(-128) + squared_parameter * (FLOAT(74) - FLOAT(47) * squared_parameter)))
    base_sigma = distances / (SEMI_MINOR * first_coefficient)
    sigma = base_sigma.copy()
    converged = numpy.zeros(len(origins), dtype=bool)
    for iteration in range(16):
        cosine_middle = numpy.cos(FLOAT(2) * first_sigma + sigma)
        sine_sigma = numpy.sin(sigma)
        cosine_sigma = numpy.cos(sigma)
        delta = second_coefficient * sine_sigma * (cosine_middle + second_coefficient / FLOAT(4) * (cosine_sigma * (FLOAT(-1) + FLOAT(2) * cosine_middle ** 2) - second_coefficient / FLOAT(6) * cosine_middle * (FLOAT(-3) + FLOAT(4) * sine_sigma ** 2) * (FLOAT(-3) + FLOAT(4) * cosine_middle ** 2)))
        updated = numpy.where(converged, sigma, base_sigma + delta)
        converged |= numpy.abs(updated - sigma) <= FLOAT(2.4e-7) * numpy.maximum(FLOAT(1), updated)
        sigma = updated
    cosine_middle = numpy.cos(FLOAT(2) * first_sigma + sigma)
    sine_sigma = numpy.sin(sigma)
    cosine_sigma = numpy.cos(sigma)
    north = sine * sine_sigma - cosine * cosine_sigma * cosine_azimuth
    second_latitude = numpy.arctan2(sine * cosine_sigma + cosine * sine_sigma * cosine_azimuth, (FLOAT(1) - FLATTENING) * numpy.hypot(sine_alpha, north))
    longitude = numpy.arctan2(sine_sigma * sine_azimuth, cosine * cosine_sigma - sine * sine_sigma * cosine_azimuth)
    correction = FLATTENING / FLOAT(16) * squared_alpha * (FLOAT(4) + FLATTENING * (FLOAT(4) - FLOAT(3) * squared_alpha))
    longitude -= (FLOAT(1) - correction) * FLATTENING * sine_alpha * (sigma + correction * sine_sigma * (cosine_middle + correction * cosine_sigma * (FLOAT(-1) + FLOAT(2) * cosine_middle ** 2)))
    result = numpy.column_stack([origins[:, 0] + longitude * RADIANS, second_latitude * RADIANS])
    result[distances == 0] = origins[distances == 0]
    arrival = numpy.arctan2(sine_alpha, -north) * RADIANS
    arrival[distances == 0] = bearings[distances == 0]
    if local_offsets:
        if numpy.any(numpy.abs(sigma) >= FLOAT(.01)) or numpy.any(numpy.abs(origins[:, 1]) >= FLOAT(85)):
            raise ValueError('local_offsets needs sub-0.01-radian nonpolar spans')
        half_sine = numpy.sin(FLOAT(.5) * sigma)
        sine_change = FLOAT(-2) * sine * half_sine * half_sine + cosine * sine_sigma * cosine_azimuth
        target_reduced = numpy.arctan2(sine * cosine_sigma + cosine * sine_sigma * cosine_azimuth, numpy.hypot(sine_alpha, north))
        reduced_change = FLOAT(2) * numpy.arcsin(numpy.clip(sine_change / (FLOAT(2) * numpy.cos(FLOAT(.5) * (reduced + target_reduced))), FLOAT(-1), FLOAT(1)))
        complement = FLOAT(1) - FLATTENING
        latitude_change = reduced_change * (numpy.cos(latitude) ** 2 + complement ** 2 * numpy.sin(latitude) ** 2) / complement
        middle_latitude = latitude + FLOAT(.5) * latitude_change
        latitude_change = reduced_change * (numpy.cos(middle_latitude) ** 2 + complement ** 2 * numpy.sin(middle_latitude) ** 2) / complement
        sine_change = FLOAT(2) * numpy.cos(latitude + FLOAT(.5) * latitude_change) * numpy.sin(FLOAT(.5) * latitude_change)
        return result, arrival, longitude * RADIANS, sine_change
    return result, arrival


def measure_city_area(polygon):
    # Nominal f32 mirror of the bounded one-span area integration. PROJ is the oracle.
    positions = numpy.array(polygon[:-1], dtype=FLOAT)
    targets = numpy.roll(positions, -1, axis=0)
    distances, bearings, _, converged = measure_inverse(positions, targets)
    if not numpy.all(converged):
        raise ValueError('city area edges must resolve')
    _, _, longitude_change, sine_change = measure_direct(positions, bearings, distances * FLOAT(.5), True, local_offsets=True)
    exact_flattening = 1 / 298.257223563
    eccentricity_squared = exact_flattening * (2 - exact_flattening)
    eccentricity = numpy.sqrt(eccentricity_squared)
    polar = 1 + (1 - eccentricity_squared) / (2 * eccentricity) * numpy.log((1 + eccentricity) / (1 - eccentricity))
    scale = FLOAT(2 * (1 - eccentricity_squared) / polar)
    radius = FLOAT(6378137 * numpy.sqrt(polar / 2))
    squared_eccentricity = FLOAT(eccentricity_squared)
    origin = positions[0]
    sine_origin = numpy.sin(origin[1] * DEGREES)

    def integrate_authalic(sine_delta):
        step = sine_delta / FLOAT(8)
        def integrand(sine):
            denominator = FLOAT(1) - squared_eccentricity * sine * sine
            return FLOAT(1) / (denominator * denominator)
        total = integrand(sine_origin) + integrand(sine_origin + sine_delta)
        for interval in range(1, 8):
            weight = FLOAT(4 if interval % 2 else 2)
            total = total + weight * integrand(sine_origin + FLOAT(interval) * step)
        return scale * total * step / FLOAT(3)

    sine_from_origin = FLOAT(2) * numpy.cos(FLOAT(.5) * (positions[:, 1] + origin[1]) * DEGREES) * numpy.sin(FLOAT(.5) * (positions[:, 1] - origin[1]) * DEGREES)
    horizontal = (positions[:, 0] - origin[0]) * DEGREES
    vertical = integrate_authalic(sine_from_origin)
    middle_horizontal = horizontal + longitude_change * DEGREES
    middle_vertical = integrate_authalic(sine_from_origin + sine_change)
    target_horizontal = numpy.roll(horizontal, -1)
    target_vertical = numpy.roll(vertical, -1)
    bulge = (middle_horizontal - horizontal) * (target_vertical - vertical) - (target_horizontal - horizontal) * (middle_vertical - vertical)
    crosses = horizontal * target_vertical - target_horizontal * vertical + (FLOAT(4) / FLOAT(3)) * bulge
    total = numpy.sum(crosses, dtype=FLOAT)
    return float(FLOAT(.5) * total * (radius * radius))


def bearing_error(actual, expected):
    difference = actual.astype(numpy.float64) - expected
    return numpy.abs(difference - 360 * numpy.rint(difference / 360))


def generate():
    randomizer = numpy.random.default_rng(48019)
    named = [
        ([20, 90], [20, 90]), ([20, 90], [-130, 90]),
        ([20, -90], [120, -90]), ([20, 90], [20, -90]),
        ([20, -90], [20, 90]), ([0, 0], [180, 0]),
        ([20, 45], [-160, -45]), ([0, 0], [179.99999, .00001]),
        ([0, 0], [179.7, .2]), ([20, 90], [30, 89.99999]),
        ([20, -90], [30, -89.99999]), ([20, 89.99999], [30, 90]),
        ([-180, 0], [180, 0]), ([179.999, 0], [-179.999, 0]),
        ([0, 0], [90, 0]), ([10, 10], [10, 10]),
    ]
    pairs = list(named)
    for index in range(512):
        first = randomizer.uniform([-180, -89.99999], [180, 89.99999])
        second = randomizer.uniform([-180, -89.99999], [180, 89.99999])
        if index < 128:
            second = first * [1, -1] + [180, 0] + randomizer.uniform([-.5, -.5], [.5, .5])
            second[0] = (second[0] + 180) % 360 - 180
        elif index < 256:
            # Cross both sides of the certification boundary at several latitudes.
            second = first * [1, -1] + [180, 0] + randomizer.uniform([-2, -2], [2, 2])
            second[0] = (second[0] + 180) % 360 - 180
            second[1] = numpy.clip(second[1], -90, 90)
        elif index < 320:
            first[1] = randomizer.choice([-90, 90])
        pairs.append((first, second))
    first = numpy.array([pair[0] for pair in pairs], dtype=FLOAT)
    second = numpy.array([pair[1] for pair in pairs], dtype=FLOAT)
    models = [('ellipsoidal', pyproj.Geod(ellps='WGS84')), ('great-circle', pyproj.Geod(a=RADIUS, f=0))]
    fixture = {'provenance': f'pyproj {pyproj.__version__}, PROJ {pyproj.proj_version_str}; seed 48019; inputs rounded to float32 before Geod.inv/fwd; arrival azimuth = back azimuth + 180 degrees.', 'pairs': [], 'direct': [], 'measurement': {}}
    sphere = models[1][1]
    for model, geodesic in models:
        initial, backward, distance = geodesic.inv(first[:, 0].astype(float), first[:, 1].astype(float), second[:, 0].astype(float), second[:, 1].astype(float))
        sphere_initial, _, sphere_distance = sphere.inv(first[:, 0].astype(float), first[:, 1].astype(float), second[:, 0].astype(float), second[:, 1].astype(float))
        midpoint_longitude, midpoint_latitude, _ = sphere.fwd(first[:, 0].astype(float), first[:, 1].astype(float), sphere_initial, sphere_distance / 2)
        for index in range(len(first)):
            fixture['pairs'].append({'model': model, 'origin': first[index].tolist(), 'target': second[index].tolist(), 'distance': float(distance[index]), 'midpoint': [float(midpoint_longitude[index]), float(midpoint_latitude[index])], 'initial': float(initial[index]), 'final': float((backward[index] + 360) % 360 - 180)})
        for origin in [[20, 90], [20, -90], [20, 89.99999], [20, -89.99999], [0, 0], [179.999, 45]]:
            for bearing in [0, 45, 90, 180, -90]:
                for length in [0, 1, 1000, 1000000, 20000000]:
                    origin = numpy.array(origin, dtype=FLOAT).astype(float)
                    longitude, latitude, arrival_backward = geodesic.fwd(*origin, bearing, length)
                    fixture['direct'].append({'model': model, 'origin': origin.tolist(), 'bearing': bearing, 'distance': length, 'destination': [longitude, latitude], 'final': (arrival_backward + 360) % 360 - 180})
        # Distances are rounded before PROJ: at a terminal pole, a fraction of a meter
        # in f32 phase can reverse arrival azimuth while position remains accurate.
        for origin, pole in [([0, 0], 90), ([0, 0], -90), ([20, 45], 90), ([20, -45], -90)]:
            bearing, _, pole_distance = geodesic.inv(*origin, origin[0], pole)
            length = float(FLOAT(pole_distance))
            longitude, latitude, arrival_backward = geodesic.fwd(*origin, bearing, length)
            fixture['direct'].append({'model': model, 'origin': origin, 'bearing': bearing, 'distance': length, 'destination': [longitude, latitude], 'final': (arrival_backward + 360) % 360 - 180})
        # Independent review regressions and neighboring paths. These include the
        # 0.0001-degree WGS84 meridian approach and the 19,900 km sphere path from 88N.
        if model == 'ellipsoidal':
            review_inputs = [([0, 0], bearing, 10001976 + offset)
                             for bearing in [.0001, -.0001, .001, -.001, .01, -.01]
                             for offset in [-300000, -100000, -10000, -1000, 0, 1000, 10000, 100000, 300000]]
        else:
            review_inputs = [([0, latitude], bearing, length)
                             for latitude in [79, -79, 80, -80, 88, -88, 89, -89]
                             for bearing in [45, 75, 135, -45, -75, -135]
                             for length in [5000000, 10000000, 19000000, 19900000, 20000000]]
            review_inputs += [([0, latitude], bearing, 5000000)
                              for latitude in [44, -44] for bearing in [0, 180]]
        for origin, bearing, length in review_inputs:
            origin = numpy.array(origin, dtype=FLOAT).astype(float)
            bearing, length = float(FLOAT(bearing)), float(FLOAT(length))
            longitude, latitude, arrival_backward = geodesic.fwd(*origin, bearing, length)
            fixture['direct'].append({'model': model, 'origin': origin.tolist(), 'bearing': bearing, 'distance': length, 'destination': [longitude, latitude], 'final': (arrival_backward + 360) % 360 - 180})
        if model == 'ellipsoidal':
            measured_distance, measured_initial, measured_final, converged = measure_inverse(first, second)
            # Coincident azimuths are conventional zeros, not PROJ's polar convention.
            nonzero = converged & (distance > 0)
            fixture['measurement']['inverse'] = {'arithmetic': 'nominal IEEE float32 (not GPU)', 'rows': len(first), 'resolved': int(numpy.sum(converged)), 'fallback': int(numpy.sum(~converged)), 'maximumDistanceErrorMeters': float(numpy.max(numpy.abs(measured_distance[converged].astype(float) - distance[converged]))), 'maximumInitialBearingErrorDegrees': float(numpy.max(bearing_error(measured_initial[nonzero], initial[nonzero]))), 'maximumFinalBearingErrorDegrees': float(numpy.max(bearing_error(measured_final[nonzero], backward[nonzero] + 180)))}
    measured_distance, measured_initial, measured_midpoint = measure_sphere(first, second)
    initial, backward, distance = sphere.inv(first[:, 0].astype(float), first[:, 1].astype(float), second[:, 0].astype(float), second[:, 1].astype(float))
    resolved = numpy.pi - distance / RADIUS >= .02
    nonzero = resolved & (distance > 0)
    midpoint_rows = numpy.array([row['midpoint'] for row in fixture['pairs'][:len(first)]])
    _, _, midpoint_errors = sphere.inv(measured_midpoint[:, 0].astype(float), measured_midpoint[:, 1].astype(float), midpoint_rows[:, 0], midpoint_rows[:, 1])
    fixture['measurement']['sphere'] = {'maximumDistanceErrorMeters': float(numpy.max(numpy.abs(measured_distance.astype(float) - distance))), 'maximumResolvedInitialBearingErrorDegrees': float(numpy.max(bearing_error(measured_initial[nonzero], initial[nonzero]))), 'maximumResolvedMidpointErrorMeters': float(numpy.max(midpoint_errors[resolved]))}
    for model, geodesic in models:
        rows = [row for row in fixture['direct'] if row['model'] == model]
        origins = numpy.array([row['origin'] for row in rows], dtype=FLOAT)
        bearings = numpy.array([row['bearing'] for row in rows], dtype=FLOAT)
        distances = numpy.array([row['distance'] for row in rows], dtype=FLOAT)
        measured, arrival = measure_direct(origins, bearings, distances, model == 'ellipsoidal')
        expected = numpy.array([row['destination'] for row in rows])
        _, _, errors = geodesic.inv(measured[:, 0].astype(float), measured[:, 1].astype(float), expected[:, 0], expected[:, 1])
        resolved_arrival = (distances > 0) & (numpy.abs(cosine_latitude(measured[:, 1])) > FLOAT(.03))
        fixture['measurement'][model + 'Direct'] = {'rows': len(rows), 'maximumPositionErrorMeters': float(numpy.max(errors)), 'maximumResolvedArrivalBearingErrorDegrees': float(numpy.max(bearing_error(arrival[resolved_arrival], numpy.array([row['final'] for row in rows])[resolved_arrival]))), 'maximumUnfilteredArrivalBearingErrorDegrees': float(numpy.max(bearing_error(arrival[distances > 0], numpy.array([row['final'] for row in rows])[distances > 0])))}
    # Legacy acceptance cases that caught over-broad numerical rewrites in the GPU gate.
    # Their strict tolerances are retained in the headless spec, independently of the
    # wider global CR01 envelope. Every coordinate is f32 before PROJ sees it.
    polar_origin = numpy.array([0, 89.9], dtype=FLOAT).astype(float)
    polar_target = numpy.array([179, 89.8], dtype=FLOAT).astype(float)
    _, _, polar_distance = sphere.inv(*polar_origin, *polar_target)
    direct_origin = numpy.array([139.7, 35.7], dtype=FLOAT).astype(float)
    direct_longitude, direct_latitude, _ = sphere.fwd(*direct_origin, 10, 250000)
    ellipsoid = models[0][1]
    line_origin = numpy.array([-74, 40.7], dtype=FLOAT).astype(float)
    line_target = numpy.array([-.13, 51.5], dtype=FLOAT).astype(float)
    _, _, line_distance = ellipsoid.inv(*line_origin, *line_target)
    polygon = numpy.array([[-.2, 51.4], [.1, 51.4], [.1, 51.6], [-.2, 51.6], [-.2, 51.4]], dtype=FLOAT).astype(float)
    area, perimeter = ellipsoid.polygon_area_perimeter(polygon[:, 0], polygon[:, 1])
    fixture['legacyRegressions'] = {
        'polarSpherePair': {'origin': polar_origin.tolist(), 'target': polar_target.tolist(), 'distance': polar_distance},
        'ordinarySphereDestination': {'origin': direct_origin.tolist(), 'bearing': 10, 'distance': 250000, 'destination': [direct_longitude, direct_latitude]},
        'ordinaryEllipsoidalLine': {'origin': line_origin.tolist(), 'target': line_target.tolist(), 'distance': line_distance},
        'cityGeodesicPolygon': {'positions': polygon.tolist(), 'area': area, 'perimeter': perimeter}
    }
    measured_area = measure_city_area(polygon)
    fixture['measurement']['cityGeodesicArea'] = {'area': measured_area, 'relativeError': abs(measured_area / area - 1)}
    print(json.dumps(fixture['measurement'], indent=2))
    output = Path(__file__).resolve().parents[1] / 'gpu-spatial-analysis/geometry-measures/cr01-geodesic-fixture.json'
    output.write_text(json.dumps(fixture, indent=2) + '\n')


if __name__ == '__main__':
    generate()
