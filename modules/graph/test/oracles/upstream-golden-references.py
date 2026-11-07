# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Prepare pinned upstream golden fixtures. No Circuitscape or PDAL binary executes here.

Expected outputs are copied from the unmodified vendored upstream files. NumPy only prepares
caller-supplied normals from original LAS positions; it never generates expected M3C2 changes.
"""
import csv
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
assert np.__version__ == '2.2.6'

root = Path(__file__).resolve().parent
upstream = root / 'upstream-golden'
manifest = json.loads((upstream / 'manifest.json').read_text())
for item in manifest['files']:
    assert hashlib.sha256((upstream / item['local']).read_bytes()).hexdigest() == item['sha256']

def numericRows(path):
    return [[float(value) for value in line.split()] for line in path.read_text().splitlines() if line.strip()]

network = upstream / 'Circuitscape.jl'
edges = numericRows(network / 'sgNetworkVerify2_graph_conductances.txt')
pairs = []
for sink, source in [(0, 1), (0, 4), (0, 6), (1, 5), (2, 6), (4, 5)]:
    voltages = numericRows(network / f'sgNetworkVerify2_voltages_{sink}_{source}.txt')
    currents = numericRows(network / f'sgNetworkVerify2_branch_currents_{sink}_{source}.txt')
    pairs.append(dict(source=source, sink=sink, voltages=voltages, absoluteCurrents=currents))
(root / 'fixtures/circuitscape-upstream-golden.json').write_text(json.dumps(dict(
    provenance='Circuitscape.jl 6.1.0 tree 4e0d29d3a7f2ed542a9f2ade782ea554c54b4ae7, unmodified test/output_verify golden outputs; Julia was not executed locally.',
    license='../upstream-golden/Circuitscape.jl/LICENSE.md', vertexCount=7, edges=edges, pairs=pairs), indent=2) + '\n')

pointcloud = upstream / 'PDAL'
def readLas(name):
    data = (pointcloud / name).read_bytes()
    assert data[:4] == b'LASF'
    offset = struct.unpack_from('<I', data, 96)[0]
    recordLength = struct.unpack_from('<H', data, 105)[0]
    count = struct.unpack_from('<I', data, 107)[0] or struct.unpack_from('<Q', data, 247)[0]
    scale = np.array(struct.unpack_from('<3d', data, 131))
    origin = np.array(struct.unpack_from('<3d', data, 155))
    return np.array([struct.unpack_from('<3i', data, offset + index * recordLength) for index in range(count)]) * scale + origin

reference = readLas('autzen-bmx-2010.las')
comparison = readLas('autzen-bmx-2023.las')
origin = np.array([194500, 259200, 400])
rows = list(csv.DictReader((pointcloud / 'autzen-bmx-m3c2.txt').open()))
cores = []
excluded = []
for index, row in enumerate(rows):
    core = np.array([float(row[axis]) for axis in ['X', 'Y', 'Z']])
    neighbors = reference[np.sum((reference - core)**2, axis=1) <= 1.390432**2]
    counts = [int(float(row['m3c2_count1'])), int(float(row['m3c2_count2']))]
    if min(counts) < 2:
        excluded.append(dict(row=index, goldenCounts=counts, reason='Golden row has no reported solution or fewer than two samples; GPU sample-variance contract requires at least two.'))
        continue
    assert len(neighbors) >= 3
    centered = neighbors - neighbors.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    normal = vectors[:, 0]
    normal *= 1 if normal[2] >= 0 else -1
    cores.append(dict(row=index, position=(core-origin).tolist(), normal=normal.tolist(),
        distance=float(row['m3c2_distance']), uncertainty=float(row['m3c2_uncertainty']),
        significant=int(float(row['m3c2_significant'])),
        populationRoughness=[float(row['m3c2_std_dev1']), float(row['m3c2_std_dev2'])], counts=counts,
        omitFirstReferencePoint=index == 0))
(root / 'fixtures/pdal-m3c2-upstream-golden.json').write_text(json.dumps(dict(
    provenance='PDAL tree 0e64af327cba0a4c54e501db0d61b62390b69a79 test/data/autzen/autzen-bmx-m3c2.txt and M3C2FilterTest.cpp verifyPoints; PDAL binary not executed locally.',
    license='../upstream-golden/PDAL/LICENSE.txt', normalPreparation='NumPy eigh of reference neighbors within normal_radius=1.390432; upward orientation. Normals are supplied inputs, not expected change outputs.',
    conventions='Recenter LAS coordinates before f32 upload. All 91 golden rows with both counts>=2 included. Row0 separately omits first reference point to reproduce the upstream filterPoints first-point self-exclusion. Upstream roughness uses population variance; GPU uses sample variance. Golden output is rounded to 0.001.',
    origin=origin.tolist(), radius=2.890432, halfDepth=5.5, reference=(reference-origin).tolist(), comparison=(comparison-origin).tolist(), cores=cores, excluded=excluded), indent=2) + '\n')
print('Circuitscape pairs:', len(pairs), 'M3C2 valid rows:', len(cores), 'excluded:', len(excluded))
