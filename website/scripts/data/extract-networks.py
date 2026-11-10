# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Rebuild the website's network fixtures. Supply the TIGER zip and Gazetteer text file."""

import argparse
import array
import csv
import heapq
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import shapefile

ROOT = Path(__file__).resolve().parents[3]
SHOWCASE = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'
OUTPUT = ROOT / 'website/src/gpu/data'


def read_column(dataset, filename, dtype):
    values = array.array(dtype)
    values.frombytes((SHOWCASE / dataset / filename).read_bytes())
    if sys.byteorder != 'little':
        values.byteswap()
    return values


def save(name, data):
    (OUTPUT / f'{name}.json').write_text(json.dumps(data, separators=(',', ':')) + '\n')


def east_north(lon, lat, center_lon, center_lat):
    # Local tangent-plane approximation. Integer meters resolve the streets and port channels.
    return (round((lon - center_lon) * 111320 * math.cos(math.radians(center_lat))),
            round((lat - center_lat) * 111320))


def streets(zip_file):
    center = (-122.405, 37.789)
    positions = {}
    adjacency = defaultdict(set)
    named = defaultdict(dict)
    for item in shapefile.Reader(zip_file).iterShapeRecords():
        record = item.record
        if record.MTFCC not in {'S1200', 'S1400'}:
            continue
        source, target = record.TNIDF, record.TNIDT
        if source == target:
            continue
        endpoints = [east_north(*point, *center) for point in
                     (item.shape.points[0], item.shape.points[-1])]
        if any(abs(x) > 2350 or abs(y) > 1500 for x, y in endpoints):
            continue
        positions[source], positions[target] = endpoints
        if record.FULLNAME:
            named[record.FULLNAME][source, target] = [east_north(*point, *center) for point in item.shape.points]
        adjacency[source].add(target)
        adjacency[target].add(source)
    # Every prefix induces a connected graph. Prioritize near-center nodes in a wide rectangle.
    def distance(node):
        x, y = positions[node]
        return max(abs(x) / 1.6, abs(y))
    origin = min(positions, key=distance)
    queue = [(distance(origin), origin)]
    ordered = {}
    while queue and len(ordered) < 700:
        _, node = heapq.heappop(queue)
        if node in ordered:
            continue
        ordered[node] = len(ordered)
        for neighbor in adjacency[node]:
            if neighbor not in ordered:
                heapq.heappush(queue, (distance(neighbor), neighbor))
    edges = sorted((index, ordered[neighbor]) for node, index in ordered.items()
                   for neighbor in adjacency[node] if neighbor in ordered and index < ordered[neighbor])
    assert len(ordered) > 256 and len(edges) > len(ordered)
    lines = []
    for segments in named.values():
        # Join consecutive TIGER edges of the same street before demonstrating line noding.
        connections = defaultdict(set)
        for source, target in segments:
            connections[source].add(target)
            connections[target].add(source)
        remaining = set(tuple(sorted(edge)) for edge in segments)
        longest = []
        while remaining:
            leaves = [node for node in connections if len(connections[node]) == 1 and
                      tuple(sorted((node, next(iter(connections[node]))))) in remaining]
            end = min(leaves) if leaves else min(remaining)[0]
            points = []
            while True:
                neighbor = next((n for n in sorted(connections[end]) if tuple(sorted((end, n))) in remaining), None)
                if neighbor is None:
                    break
                remaining.remove(tuple(sorted((end, neighbor))))
                path = segments.get((end, neighbor)) or list(reversed(segments[neighbor, end]))
                points.extend(path if not points else path[1:])
                end = neighbor
            if path_length(points) > path_length(longest):
                longest = points
        if len(longest) >= 2:
            lines.append(longest)
    lines = sorted(lines, key=path_length, reverse=True)[:16]
    save('sf-market-streets', {'nodes': [positions[node] for node in ordered], 'edges': edges,
                               'lines': [resample_line(line) for line in lines]})


def path_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def resample_line(line):
    result = [line[0]]
    cumulative = 0
    total = path_length(line)
    for first, second in zip(line, line[1:]):
        length = math.dist(first, second)
        while len(result) < 8 and cumulative + length >= len(result) * total / 8:
            fraction = (len(result) * total / 8 - cumulative) / length
            result.append([round(a + (b - a) * fraction) for a, b in zip(first, second)])
        cumulative += length
    return result + [line[-1]]



def vessel_tracks():
    dataset = 'poopdeck-ais-us'
    vertices = read_column(dataset, 'vertices.bin', 'f')
    offsets = read_column(dataset, 'pathOffsets.bin', 'I')
    times = read_column(dataset, 'time.bin', 'I')
    vessels = read_column(dataset, 'vesselIndex.bin', 'I')
    candidates = {}
    for track in range(len(vessels)):
        fixes = []
        # Only contiguous fixes in the window. A vessel leaving and reentering starts a new leg.
        for row in range(offsets[track], offsets[track + 1] + 1):
            if row < offsets[track + 1]:
                lon, lat = vertices[row * 2:row * 2 + 2]
                if -122.595 < lon < -122.20 and 37.69 < lat < 37.92:
                    if not fixes or times[row] > fixes[-1][2]:
                        fixes.append((*east_north(lon, lat, -122.40, 37.80), times[row]))
                    continue
            if len(fixes) >= 8:
                extent = math.hypot(max(x for x, _, _ in fixes) - min(x for x, _, _ in fixes),
                                    max(y for _, y, _ in fixes) - min(y for _, y, _ in fixes))
                score = extent * math.sqrt(len(fixes))
                if score > candidates.get(vessels[track], (0, []))[0]:
                    candidates[vessels[track]] = (score, fixes)
            fixes = []
    ranked = sorted(candidates.values(), reverse=True)
    selected = [fixes for score, fixes in ranked[:40]]
    assert len(selected) == 40
    # Round-robin between long transits and local traffic so small demos include both.
    selected = [selected[index] for low in range(10) for index in range(low, 40, 10)]
    save('sf-bay-ais', selected)


def commute_flows(gazetteer):
    geoids = json.loads((SHOWCASE / 'chicago-tracts/manifest.json').read_text())['geoid']
    with open(gazetteer) as source:
        centers = {r['GEOID']: (float(r['INTPTLONG']), float(r['INTPTLAT']))
                   for r in csv.DictReader((line.rstrip() for line in source), delimiter='\t')}
    dataset = 'chicago-lodes-od'
    origins, destinations, counts = (read_column(dataset, column + '.bin', 'I')
                                     for column in ('origin', 'destination', 'count'))
    # Sample origins and workplaces in central Chicago, excluding suburb-to-city commuting.
    points = []
    node_ids = {}
    flows = []
    for origin, destination, jobs in zip(origins, destinations, counts):
        endpoints = [centers.get(str(geoids[index])) for index in (origin, destination)]
        if any(p is None or not (-87.74 < p[0] < -87.605 and 41.84 < p[1] < 41.935) for p in endpoints):
            continue
        pair = []
        for point in endpoints:
            if point not in node_ids:
                node_ids[point] = len(points)
                points.append(east_north(*point, -87.67, 41.89))
            pair.append(node_ids[point])
        flows.append([*pair, jobs])
        if len(flows) == 1200:
            break
    assert len(flows) == 1200
    save('chicago-commutes', {'tracts': points, 'flows': flows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tiger_zip')
    parser.add_argument('gazetteer')
    args = parser.parse_args()
    streets(args.tiger_zip)
    vessel_tracks()
    commute_flows(args.gazetteer)
