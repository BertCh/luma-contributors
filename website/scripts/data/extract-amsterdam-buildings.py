# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Extract Amsterdam 3DBAG footprints and relative roof heights from a CityJSON response.

Source, bounding box and required credit appear in src/gpu/data/area-sources.md.
Pass the downloaded CityJSON feature collection as the sole argument.
"""
import json
from pathlib import Path
import sys

collection = json.loads(Path(sys.argv[1]).read_text())
transform = collection['metadata']['transform']
origin = [119400, 485942]
buildings = []
for feature in collection['features']:
    building = next(obj for obj in feature['CityObjects'].values() if obj['type'] == 'Building')
    footprint = next(geometry for geometry in building['geometry'] if geometry['lod'] == '0')
    if len(footprint['boundaries']) != 1 or len(footprint['boundaries'][0]) != 1:
        continue
    attributes = building['attributes']
    roof = attributes.get('b3_h_dak_70p')
    ground = attributes.get('b3_h_maaiveld')
    if roof is None or ground is None or roof <= ground:
        continue
    # Local RD metres, centimetre quantization. Shared vertices receive identical coordinates.
    ring = [
        [round(feature['vertices'][index][axis] * transform['scale'][axis] +
               transform['translate'][axis] - origin[axis], 2) for axis in [0, 1]]
        for index in footprint['boundaries'][0][0]
    ]
    buildings.append({
        'id': feature['id'].removeprefix('NL.IMBAG.Pand.'),
        'ring': ring,
        'height': round(roof - ground, 2)
    })
destination = Path(__file__).resolve().parents[2] / 'src/gpu/data/amsterdam-buildings.json'
buildings.sort(key=lambda building: building['id'])
destination.write_text(json.dumps(buildings, separators=(',', ':')) + '\n')
