# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent SciPy interpolation/NCC and NetworkX quotient-graph merge fixtures."""
import json
from pathlib import Path
import numpy
import scipy
from scipy.ndimage import map_coordinates
import networkx


def correlation(reference, moving, reference_mask, moving_mask, shift):
    rows, columns = numpy.indices(reference.shape)
    coordinates = [rows - shift[1], columns - shift[0]]
    sampled = map_coordinates(moving, coordinates, order=1, mode='constant', cval=0.0, prefilter=False)
    support = map_coordinates(moving_mask.astype(float), coordinates, order=1, mode='constant', cval=0.0, prefilter=False)
    valid = reference_mask & (support >= 1.0 - 1e-12)
    count = int(valid.sum())
    if count < max(3, numpy.ceil(0.3 * min(reference_mask.sum(), moving_mask.sum()))):
        return None, count
    first = reference[valid] - reference[valid].mean()
    second = sampled[valid] - sampled[valid].mean()
    denominator = numpy.linalg.norm(first) * numpy.linalg.norm(second)
    return (float(first @ second / denominator) if denominator > 1e-12 else None), count


def registration(translation):
    width, height = 16, 12
    rows, columns = numpy.indices((height, width), dtype=float)
    def image(column, row):
        return (3 * numpy.exp(-((column - 4)**2 + (row - 3)**2) / 5)
                + 2 * numpy.exp(-((column - 11)**2 + (row - 7)**2) / 8)
                + numpy.sin(column * 0.7 + row * 0.2) * 0.3 + numpy.cos(row * 0.9) * 0.2)
    reference = image(columns, rows)
    moving = image(columns - translation[0], rows - translation[1])
    reference_mask = (columns.astype(int) * 5 + rows.astype(int) * 3) % 13 != 0
    moving_mask = (columns.astype(int) * 3 + rows.astype(int) * 7) % 17 != 0
    reference_mask[3:5, 9:12] = False
    moving_mask[6:8, 2:5] = False
    candidates = []
    for vertical in range(-3, 4):
        for horizontal in range(-4, 5):
            score, count = correlation(reference, moving, reference_mask, moving_mask, [horizontal, vertical])
            if score is not None:
                candidates.append((score, horizontal, vertical, count))
    coarse = max(candidates, key=lambda value: value[0])
    candidates = []
    for vertical in range(-10, 11):
        for horizontal in range(-10, 11):
            shift = [coarse[1] + horizontal / 20, coarse[2] + vertical / 20]
            if abs(shift[0]) > 4 or abs(shift[1]) > 3:
                continue
            score, count = correlation(reference, moving, reference_mask, moving_mask, shift)
            if score is not None:
                candidates.append((score, -(horizontal**2 + vertical**2), *shift, count))
    selected = max(candidates)
    return {'width': width, 'height': height, 'reference': reference.flatten().tolist(), 'moving': moving.flatten().tolist(),
            'referenceMask': reference_mask.astype(int).flatten().tolist(), 'movingMask': moving_mask.astype(int).flatten().tolist(),
            'expectedShift': list(selected[2:4]), 'expectedPeak': selected[0], 'expectedOverlap': selected[4], 'translation': translation}


def regions(values, labels, threshold, rounds):
    graph = networkx.Graph()
    for label in sorted(set(labels)):
        graph.add_node(label, sources={label}, samples=[value for value, source in zip(values, labels) if source == label])
    for first, second in zip(labels[:-1], labels[1:]):
        if first != second:
            graph.add_edge(first, second)
    for _ in range(rounds):
        candidates = sorted((abs(numpy.mean(graph.nodes[first]['samples']) - numpy.mean(graph.nodes[second]['samples'])), min(first, second), max(first, second)) for first, second in graph.edges)
        if not candidates or candidates[0][0] > threshold:
            break
        _, first, second = candidates[0]
        graph.nodes[first]['samples'] += graph.nodes[second]['samples']
        graph.nodes[first]['sources'] |= graph.nodes[second]['sources']
        neighbors = list(graph.neighbors(second))
        graph.remove_node(second)
        graph.add_edges_from((first, neighbor) for neighbor in neighbors if neighbor != first)
    pending = any(abs(numpy.mean(graph.nodes[first]['samples']) - numpy.mean(graph.nodes[second]['samples'])) <= threshold for first, second in graph.edges)
    mapping = [next(root for root in graph.nodes if label in graph.nodes[root]['sources']) for label in range(max(labels) + 1)]
    return {'values': values, 'labels': labels, 'threshold': threshold, 'rounds': rounds, 'expectedMapping': mapping, 'expectedStatus': int(pending)}

fixture = {'versions': {'numpy': numpy.__version__, 'scipy': scipy.__version__, 'networkx': networkx.__version__},
    'registration': [registration([2, -1]), registration([1.35, -0.65])],
    'regions': [regions([0, 2, 4], [0, 1, 2], 2.1, 3), regions([0, 0, 0, 2, 4], [0, 0, 0, 1, 2], 3.2, 3), regions([0, 1, 2], [0, 1, 2], 2, 1), regions([0, 1, 2], [0, 1, 2], 2, 2)]}
Path(__file__).with_name('fixtures').joinpath('raster-registration-regions.json').write_text(json.dumps(fixture, indent=2) + '\n')
