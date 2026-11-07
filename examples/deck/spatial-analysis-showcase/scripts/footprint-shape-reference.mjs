// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile, writeFile} from 'node:fs/promises';
import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';

// GEOS supplies an independent geometry oracle. Only the declared display-frame conversion
// is reproduced here so CPU and GPU measures refer to identical float32 planar coordinates.
const python = process.argv[2] ?? 'python3';
const outputFile = process.argv[3] ?? '/tmp/footprint-shape-reference.json';
const collection = JSON.parse(await readFile(new URL('../public/data/chicago-buildings/buildings.geojson', import.meta.url), 'utf8'));
const inputs = [];
for (const sourceId of [354712, 814558]) {
  const feature = collection.features.find(item => Number(item.properties.bldg_id) === sourceId);
  assert.ok(feature);
  const polygons = feature.geometry.type === 'Polygon' ? [feature.geometry.coordinates] : feature.geometry.coordinates;
  const shell = polygons[0][0];
  const origin = getRingCentroid(shell);
  const scales = getDistanceScales({longitude: origin[0], latitude: origin[1], highPrecision: true});
  const metersPerDegree = scales.unitsPerDegree.map((value, axis) => value * scales.metersPerUnit[axis]);
  const project = ([longitude, latitude]) => {
    let horizontal = (longitude - origin[0]) * metersPerDegree[0];
    let vertical = (latitude - origin[1]) * metersPerDegree[1];
    const estimated = addMetersToLngLat(origin, [horizontal, vertical]);
    horizontal += (longitude - estimated[0]) * metersPerDegree[0];
    vertical += (latitude - estimated[1]) * metersPerDegree[1];
    return [Math.fround(horizontal), Math.fround(vertical)];
  };
  inputs.push({sourceId, origin, polygons: polygons.map(polygon => polygon.map(ring => ring.map(project)))});
}
function getRingCentroid(ring) {
  const uniqueCoordinates = ring.length > 1 ? ring.slice(0, -1) : ring;
  let twiceArea = 0;
  let longitudeSum = 0;
  let latitudeSum = 0;
  for (let index = 0; index < uniqueCoordinates.length; index++) {
    const coordinate = uniqueCoordinates[index];
    const nextCoordinate = uniqueCoordinates[(index + 1) % uniqueCoordinates.length];
    const cross = coordinate[0] * nextCoordinate[1] - nextCoordinate[0] * coordinate[1];
    twiceArea += cross;
    longitudeSum += (coordinate[0] + nextCoordinate[0]) * cross;
    latitudeSum += (coordinate[1] + nextCoordinate[1]) * cross;
  }
  if (twiceArea === 0) {
    return [
      uniqueCoordinates.reduce((sum, coordinate) => sum + coordinate[0], 0) / uniqueCoordinates.length,
      uniqueCoordinates.reduce((sum, coordinate) => sum + coordinate[1], 0) / uniqueCoordinates.length
    ];
  }
  return [longitudeSum / (3 * twiceArea), latitudeSum / (3 * twiceArea)];
}
const program = `import json,sys,shapely
from shapely.geometry import Polygon,MultiPolygon,LineString
rows=json.load(sys.stdin)
output=[]
for row in rows:
 polygons=[Polygon(rings[0],rings[1:]) for rings in row['polygons']]
 geometry=MultiPolygon(polygons)
 circle=shapely.maximum_inscribed_circle(geometry,tolerance=.0001)
 witness=shapely.minimum_clearance_line(geometry)
 buffers=[]
 for distance in [-4,4]:
  for join in ['round','mitre','bevel']:
   value=shapely.buffer(geometry,distance,quad_segs=8,join_style=join,mitre_limit=5)
   buffers.append({'distance':distance,'join':join,'area':value.area,'empty':value.is_empty,'valid':value.is_valid})
 output.append({'sourceId':row['sourceId'],'origin':row['origin'],'area':geometry.area,'valid':geometry.is_valid,'circleRadius':circle.length,'circleWitness':list(circle.coords),'minimumClearance':shapely.minimum_clearance(geometry),'clearanceWitness':list(witness.coords),'buffers':buffers,'inputPolygons':row['polygons']})
print(json.dumps({'shapelyVersion':shapely.__version__,'geosVersion':shapely.geos_version_string,'references':output}))`;
const reference = JSON.parse(execFileSync(python, ['-c', program], {input: JSON.stringify(inputs), encoding: 'utf8'}));
assert.ok(reference.references.every(item => item.valid && item.circleRadius > 0 && item.minimumClearance > 0));
await writeFile(outputFile, JSON.stringify(reference, null, 2) + '\n');
console.log(JSON.stringify(reference.references.map(({inputPolygons, ...item}) => item)));
