# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""PROJ Mercator oracle on packed float32 inputs. Run with /usr/bin/python3."""

import json
import math
import struct
from pathlib import Path
from pyproj import Transformer


def packFloat(value):
    return struct.unpack('f', struct.pack('f', value))[0]


transformer = Transformer.from_crs(4326, 3857, always_xy=True)
halfExtent = math.pi * 6378137
latitudeLimit = packFloat(85.051129)
fixtures = []
positions = [(-122.4194, 37.7749), (10.40744, 57.64911), (151.2093, -33.8688),
             (-70.6693, -33.4489), (77.209, 28.6139), (132.25, 72.5),
             (-179.99999, 83.75), (179.99999, -83.75), (0, 0), (-180, 0),
             (180, 0), (-200, 100), (200, -100)]
positions.extend(((index * 137.50776405) % 360 - 180, (index * 47.1351) % 170 - 85) for index in range(100))
for longitude, latitude in positions:
    longitude, latitude = packFloat(longitude), packFloat(latitude)
    projectedFirst, projectedSecond = transformer.transform(max(-180, min(180, longitude)),
                                                           max(-latitudeLimit, min(latitudeLimit, latitude)))
    for level in [0, 1, 15, 26, 29]:
        size = 2 ** level
        column = math.floor((projectedFirst + halfExtent) / (2 * halfExtent) * size) % size
        row = max(0, min(size - 1, math.floor((halfExtent - projectedSecond) / (2 * halfExtent) * size)))
        fixtures.append(dict(longitude=longitude, latitude=latitude, level=level, column=column, row=row))

Path(__file__).with_suffix('.json').write_text(json.dumps(fixtures, indent=2) + '\n')
