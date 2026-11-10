# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Prepare geographically aligned Chicago block job counts and tract commute lines.

Pass the directory containing the two LODES8 Illinois downloads listed in area-sources.md.
"""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

WEBSITE = Path(__file__).resolve().parents[2]
OUTPUT = WEBSITE / 'src/gpu/data'
ARCHIVE = WEBSITE.parent / 'examples/deck/spatial-analysis-showcase/public/data'
DOWNLOADS = Path(sys.argv[1])
tracts = json.loads((OUTPUT / 'chicago-tracts.json').read_text())
geoids = [tract['geoid'] for tract in tracts]
# The zonal run uses an eight-by-eight equivalent window.
zone_ids = set(geoids[:64])


def rows(filename):
    with gzip.open(DOWNLOADS / filename, 'rt') as source:
        yield from csv.DictReader(source)


coordinates = {
    row['tabblk2020']: [round((float(row['blklondd']) + 87.67) * 111195 * math.cos(math.radians(41.878))),
                        round((float(row['blklatdd']) - 41.878) * 111195)]
    for row in rows('il_xwalk.csv.gz') if row['trct'] in zone_ids
}
blocks = {
    row['w_geocode']: [*coordinates[row['w_geocode']], int(row['C000']), geoids.index(row['w_geocode'][:11])]
    for row in rows('il_wac_S000_JT00_2021.csv.gz') if row['w_geocode'] in coordinates
}
# An order independent of block geography keeps slider prefixes distributed across the window.
order = sorted(blocks, key=lambda geoid: hashlib.sha256(geoid.encode()).hexdigest())
(OUTPUT / 'chicago-workplace-blocks.json').write_text(json.dumps(
    [blocks[geoid] for geoid in order], separators=(',', ':')) + '\n')

all_geoids = json.loads((ARCHIVE / 'chicago-tracts/manifest.json').read_text())['geoid']
index_map = {index: geoids.index(geoid) for index, geoid in enumerate(all_geoids) if geoid in geoids}


def integers(filename):
    return [value for (value,) in struct.iter_unpack('<I', (ARCHIVE / 'chicago-lodes-od' / filename).read_bytes())]


# Keep the strongest 40 home/work pairs for each window size shown by the control.
flows = [(index_map[start], index_map[end], jobs) for start, end, jobs in
         zip(integers('origin.bin'), integers('destination.bin'), integers('count.bin'))
         if start in index_map and end in index_map]
flows.sort(key=lambda flow: -flow[2])
selected = set()
for side in range(4, 13):
    eligible = [flow for flow in flows if max(flow[:2]) < side * side]
    selected.update(eligible[:40])
assert all(sum(max(flow[:2]) < side * side for flow in selected) >= 40 for side in range(4, 13))
(OUTPUT / 'chicago-commute-lines.json').write_text(json.dumps(
    [flow for flow in flows if flow in selected], separators=(',', ':')) + '\n')
