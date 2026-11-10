# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Refresh the small geometry samples. Requires Python 3 and Shapely 2."""

import json
import math
from pathlib import Path
from urllib.request import urlopen

from shapely import affinity
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import polylabel

BASE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/"
OUTPUT = Path(__file__).resolve().parents[1] / "src/gpu/data/natural-earth-geometry.json"


def read_features(layer):
    with urlopen(BASE_URL + layer + ".geojson", timeout=60) as source:
        return json.load(source)["features"]


def rounded(points):
    return [[round(x, 5), round(y, 5)] for x, y in points]


def shoreline_profile(polygon):
    # Work in a local equirectangular plane. Each ray stops at the first shore.
    # The positive radii ensure a simple open ring with a known interior fan origin.
    polygon = affinity.scale(polygon, xfact=math.cos(math.radians(polygon.centroid.y)))
    center = polylabel(polygon, tolerance=0.0005)
    radii = []
    for index in range(64):
        angle = 2 * math.pi * index / 64
        ray = LineString([
            (center.x, center.y),
            (center.x + 50 * math.cos(angle), center.y + 50 * math.sin(angle))
        ])
        radii.append(center.distance(polygon.boundary.intersection(ray)))
    longest_radius = max(radii)
    return [round(radius / longest_radius, 4) for radius in radii]


def convex_profile(shore):
    """Inscribed CCW hexagon made of six measured vertices of the convex shoreline hull."""
    longitude_scale = math.cos(math.radians(sum(y for _, y in shore) / len(shore)))
    points = sorted(set((x * longitude_scale, y) for x, y in shore))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def half_hull(ordered):
        hull = []
        for point in ordered:
            while len(hull) > 1 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
        return hull

    hull = half_hull(points)[:-1] + half_hull(reversed(points))[:-1]
    while len(hull) > 6:
        smallest = min(range(len(hull)), key=lambda i: cross(hull[i - 1], hull[i], hull[(i + 1) % len(hull)]))
        hull.pop(smallest)
    x = sum(x for x, _ in hull) / len(hull)
    y = sum(y for _, y in hull) / len(hull)
    radius = max(math.hypot(east - x, north - y) for east, north in hull)
    return [[round((east - x) / radius, 5), round((north - y) / radius, 5)] for east, north in hull]


def generalized_coastlines(features):
    lines = []
    for feature in features:
        line = shape(feature["geometry"])
        west, south, east, north = line.bounds
        if max(east - west, north - south) >= 4:
            simplified = line.simplify(0.9, preserve_topology=False)
            lines.append([[round(x, 1), round(y, 1)] for x, y in simplified.coords])
    return lines


def greenland_ring(coastline):
    # Greenland is the one closed coast spanning this interior Arctic bounding box.
    for ring in coastline:
        x, y = zip(*ring)
        if ring[0] == ring[-1] and min(x) < -63 < -26 < max(x) and min(y) < 65 < 81 < max(y) and max(x) < 0:
            points = ring[:-1]
            area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
            return points if area > 0 else list(reversed(points))
    raise ValueError("Greenland shell missing")


def prepare():
    coastline = generalized_coastlines(read_features("ne_110m_coastline"))
    lakes = read_features("ne_10m_lakes")
    lake_names = ["Lake Superior", "Lake Michigan", "Lake Huron", "Lake Erie", "Lake Ontario"]
    profiles = [
        "Lake Erie", "Lake Ontario", "Lake Tahoe", "Lake Okeechobee", "Lake Winnebago",
        "Lake Winnipeg", "Lake Nipissing", "Lake Claire"
    ]
    polygons = {
        feature["properties"]["name"]: shape(feature["geometry"])
        for feature in lakes if feature["properties"]["name"] in lake_names + profiles
    }
    # The spatial joins operate on open shell rings. Omit islands and generalize to 0.055 degrees.
    shores = {
        name: rounded(Polygon(polygons[name].exterior).simplify(0.055).exterior.coords[:-1])
        for name in lake_names
    }
    rivers = read_features("ne_10m_rivers_lake_centerlines")
    reaches = {}
    for name in ["Missouri", "Ohio", "Columbia", "Snake", "Mississippi", "Mackenzie"]:
        parts = [
            line for feature in rivers if feature["properties"]["name"] == name
            for line in shape(feature["geometry"]).geoms
        ]
        longest = max(parts, key=lambda line: line.length)
        reaches[name] = rounded(longest.simplify(0.018).coords)
    cities = read_features("ne_110m_populated_places_simple")
    names = [
        "New York", "London", "Los Angeles", "Chicago", "Paris", "São Paulo", "Buenos Aires",
        "Cape Town", "Johannesburg", "Nairobi", "New Delhi", "Beijing", "Tokyo",
        "Singapore", "Jakarta", "Sydney", "Cairo", "Mexico City"
    ]
    places = {
        feature["properties"]["name"]: rounded([feature["geometry"]["coordinates"]])[0]
        for feature in cities if feature["properties"]["name"] in names
    }
    data = {
        "greatLakes": shores,
        "convexShores": {name: convex_profile(shore) for name, shore in shores.items()},
        "shoreRadii": {name: shoreline_profile(polygons[name]) for name in profiles},
        "rivers": reaches,
        "cities": places,
        "greenlandShore": greenland_ring(coastline)
    }
    OUTPUT.with_name("world-coastline.json").write_text(
        json.dumps(coastline, separators=(",", ":")) + "\n"
    )
    OUTPUT.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    prepare()
