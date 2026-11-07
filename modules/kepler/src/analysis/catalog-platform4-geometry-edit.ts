// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUAffineTransform,
  GPUGeodesicDestination,
  GPUGeodesicPairs,
  GPUGeometryOrientation,
  getGPUAffineTransformParameterValues,
  getGPUGeometryOrientationParameterValues,
  type GPUSpatialContext
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {
  EARTH_RADIUS_METERS,
  addCatalogDiagnostic,
  appendCatalogRow,
  getPointPositions,
  getPolygonParts,
  makeCatalogGeometryText,
  makeCatalogTable,
  measureRingSignedArea,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogGeometry,
  type CatalogPosition
} from './catalog-geojson';
import {normalizeLongitude} from './catalog-geodesy';
import {packCatalogPolygons} from './catalog-gpu-pack';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  collectCatalogPositions,
  getCatalogSourceColumns
} from './catalog-platform4-geometry-common';
import {makeLocalFrame} from './catalog-local-frame';
import {mapGeometryCoordinates} from './catalog-geometry-utils';

/**
 * GPUGeometryOrientation (orient-polygons). The GPU decides which rings wind the wrong way by an
 * exact orientation test at the extreme vertex; reversal is applied to the float64 source rings, so
 * coordinates are bit-exact. Rings with zero area are reported like the CPU reference. A
 * self-intersecting ring is judged at its extreme vertex on the GPU and by signed area on the CPU.
 */
export async function runGpuGeometryOrientation(
  context: CatalogGpuContext,
  polygons: CatalogFeatureCollection,
  parameters: {shellOrientation: 'counterclockwise' | 'clockwise'}
): Promise<CatalogRunOutput> {
  const packed = packCatalogPolygons(polygons.features, {closing: 'drop'});
  if (!packed.ringCount) throw new CatalogGpuUnsupportedError('No polygon feature');
  const clockwise = parameters.shellOrientation === 'clockwise';
  const reversedRings = await withCatalogGpuSession(
    context.device,
    'catalog-geometry-orientation',
    async session => {
      const positions = session.output('oriented-positions', 'float32x2', packed.vertexCount);
      const reversed = session.output('reversed-rings', 'uint32', packed.ringCount);
      session.graph.add(
        new GPUGeometryOrientation({
          mode: 'orient-polygons',
          positions: session.input('positions', packed.positions, 'float32x2'),
          featureOffsets: session.input('feature-offsets', packed.featureOffsets, 'uint32'),
          polygonOffsets: session.input('polygon-offsets', packed.polygonOffsets, 'uint32'),
          ringOffsets: session.input('ring-offsets', packed.ringOffsets, 'uint32'),
          parameters: session.input(
            'parameters',
            getGPUGeometryOrientationParameterValues({exteriorClockwise: clockwise}),
            'float32'
          ),
          output: {positions: positions.view, reversedRings: reversed.view}
        })
      );
      session.run();
      return session.readUint32(reversed);
    }
  );
  const table = makeCatalogTable(
    'oriented-polygons',
    'Oriented polygons',
    {source_index: 'int32', source_id: 'string', rings_reversed: 'int32', geometry: 'string'},
    'geometry'
  );
  let ringRow = 0;
  let zeroArea = 0;
  let reversedTotal = 0;
  polygons.features.forEach((feature, sourceIndex) => {
    let reversed = 0;
    const oriented = getPolygonParts(feature.geometry).map(polygon =>
      polygon.map(ring => {
        const flipped = reversedRings[ringRow++] === 1;
        if (measureRingSignedArea(ring, 'planar') === 0) zeroArea++;
        if (!flipped) return ring;
        reversed++;
        return [...ring].reverse();
      })
    );
    reversedTotal += reversed;
    const isMulti = feature.geometry?.type === 'MultiPolygon';
    appendCatalogRow(table, {
      ...getCatalogSourceColumns(feature, sourceIndex),
      rings_reversed: reversed,
      geometry: isMulti
        ? makeCatalogGeometryText('MultiPolygon', oriented)
        : makeCatalogGeometryText('Polygon', oriented[0] ?? [])
    });
  });
  const diagnostics: CatalogDiagnostic[] = [];
  if (zeroArea)
    addCatalogDiagnostic(
      diagnostics,
      'zero-area-ring',
      'warning',
      'Rings with zero planar area have no orientation and are left unchanged.',
      zeroArea
    );
  return {
    tables: [table],
    diagnostics,
    summary: {ringsReversed: reversedTotal},
    dispatchedContributors: ['GPUGeometryOrientation']
  };
}

/**
 * GPUAffineTransform with one global matrix. Every vertex maps independently, so all vertices of the
 * collection run as one point set. Positions are shifted by the pivot in float64 first so the
 * float32 matrix is applied to small offsets; the CPU applies scale, then rotation about the pivot,
 * then translation, which is the contributor's matrix order.
 */
export async function runGpuAffineTransform(
  context: CatalogGpuContext,
  features: CatalogFeatureCollection,
  parameters: {
    translateX: number;
    translateY: number;
    scaleX: number;
    scaleY: number;
    rotationDegrees: number;
    pivot: 'origin' | 'bounds-center';
  }
): Promise<CatalogRunOutput> {
  const allPositions = collectCatalogPositions([features]);
  if (!allPositions.length) throw new CatalogGpuUnsupportedError('No vertices to transform');
  let pivotX = 0;
  let pivotY = 0;
  if (parameters.pivot === 'bounds-center') {
    const frame = makeLocalFrame('planar', allPositions, {shift: 'center'});
    [pivotX, pivotY] = frame.inverse(0, 0);
  }
  // Vertices in the order the geometry mapper visits them.
  const vertices: CatalogPosition[] = [];
  const visit = (geometry: CatalogGeometry) => {
    if (geometry)
      mapGeometryCoordinates(geometry.coordinates, position => (vertices.push(position), position));
  };
  for (const feature of features.features) visit(feature.geometry);
  const shifted = new Float32Array(vertices.length * 2);
  vertices.forEach((position, index) => {
    shifted[2 * index] = position[0] - pivotX;
    shifted[2 * index + 1] = position[1] - pivotY;
  });
  const transformed = await withCatalogGpuSession(
    context.device,
    'catalog-affine-transform',
    async session => {
      const output = session.output('transformed', 'float32x2', vertices.length);
      session.graph.add(
        new GPUAffineTransform({
          geometryType: 'points',
          positions: session.input('positions', shifted, 'float32x2'),
          parameters: session.input(
            'parameters',
            getGPUAffineTransformParameterValues({
              scale: [parameters.scaleX, parameters.scaleY],
              rotate: parameters.rotationDegrees,
              translate: [parameters.translateX, parameters.translateY],
              origin: [0, 0]
            }),
            'float32'
          ),
          output: {positions: output.view}
        })
      );
      session.run();
      return session.readFloat32(output);
    }
  );
  const table = makeCatalogTable(
    'transformed-features',
    'Transformed features',
    {source_index: 'int32', source_id: 'string', geometry: 'string'},
    'geometry'
  );
  let cursor = 0;
  let empty = 0;
  features.features.forEach((feature, sourceIndex) => {
    if (!feature.geometry) {
      empty++;
      return;
    }
    const geometryText = makeCatalogGeometryText(
      feature.geometry.type,
      mapGeometryCoordinates(feature.geometry.coordinates, () => {
        const index = cursor++;
        return [pivotX + transformed[2 * index], pivotY + transformed[2 * index + 1]];
      })
    );
    appendCatalogRow(table, {
      ...getCatalogSourceColumns(feature, sourceIndex),
      geometry: geometryText
    });
  });
  const diagnostics: CatalogDiagnostic[] = [];
  if (empty)
    addCatalogDiagnostic(
      diagnostics,
      'empty-geometry',
      'warning',
      'Features without geometry were skipped.',
      empty
    );
  return {
    tables: [table],
    diagnostics,
    summary: {featureCount: table.rowCount},
    dispatchedContributors: ['GPUAffineTransform']
  };
}

function getSphereContext(): GPUSpatialContext {
  return {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters',
    sphereRadius: EARTH_RADIUS_METERS
  };
}

/** GPUGeodesicDestination on the sphere; float32 output gives about 1e-5 degrees of longitude. */
export async function runGpuGeodesicDestination(
  context: CatalogGpuContext,
  points: CatalogFeatureCollection,
  parameters: {bearing: number; distance: number; coordinateSystem: CatalogCoordinateSystem}
): Promise<CatalogRunOutput> {
  if (parameters.coordinateSystem === 'planar')
    throw new CatalogGpuUnsupportedError('GPUGeodesicDestination needs longitude/latitude');
  const origins: {position: CatalogPosition; sourceIndex: number}[] = [];
  points.features.forEach((feature, sourceIndex) => {
    for (const position of getPointPositions(feature.geometry))
      origins.push({position, sourceIndex});
  });
  if (!origins.length) throw new CatalogGpuUnsupportedError('No points');
  const destinations = await withCatalogGpuSession(
    context.device,
    'catalog-geodesic-destination',
    async session => {
      const output = session.output('destinations', 'float32x2', origins.length);
      session.graph.add(
        new GPUGeodesicDestination({
          spatialContext: getSphereContext(),
          origins: session.input(
            'origins',
            Float32Array.from(origins.flatMap(({position}) => [position[0], position[1]])),
            'float32x2'
          ),
          bearings: session.input(
            'bearings',
            new Float32Array(origins.length).fill(parameters.bearing),
            'float32'
          ),
          distances: session.input(
            'distances',
            new Float32Array(origins.length).fill(parameters.distance),
            'float32'
          ),
          output: {destinations: output.view}
        })
      );
      session.run();
      return session.readFloat32(output);
    }
  );
  const table = makeCatalogTable(
    'geodesic-destinations',
    'Destination points',
    {
      destination_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      bearing: 'float64',
      distance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  origins.forEach(({sourceIndex}, row) =>
    appendCatalogRow(table, {
      destination_id: `d${table.rowCount}`,
      ...getCatalogSourceColumns(points.features[sourceIndex], sourceIndex),
      bearing: parameters.bearing,
      distance: parameters.distance,
      geometry: makeCatalogGeometryText('Point', [
        normalizeLongitude(destinations[2 * row]),
        destinations[2 * row + 1]
      ])
    })
  );
  return {
    tables: [table],
    diagnostics: [],
    summary: {destinationCount: table.rowCount},
    dispatchedContributors: ['GPUGeodesicDestination']
  };
}

/** GPUGeodesicPairs on the sphere; distances and bearings are float32 (about 1e-6 relative). */
export async function runGpuGeodesicPairs(
  context: CatalogGpuContext,
  origins: CatalogFeatureCollection,
  destinations: CatalogFeatureCollection,
  parameters: {coordinateSystem: CatalogCoordinateSystem}
): Promise<CatalogRunOutput> {
  if (parameters.coordinateSystem === 'planar')
    throw new CatalogGpuUnsupportedError('GPUGeodesicPairs needs longitude/latitude');
  const flatten = (collection: CatalogFeatureCollection) =>
    collection.features.flatMap(feature => getPointPositions(feature.geometry));
  const from = flatten(origins);
  const to = flatten(destinations);
  const pairCount = Math.min(from.length, to.length);
  if (!pairCount) throw new CatalogGpuUnsupportedError('No point pairs');
  const pack = (positions: readonly CatalogPosition[]) =>
    Float32Array.from(
      positions.slice(0, pairCount).flatMap(position => [position[0], position[1]])
    );
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-geodesic-pairs',
    async session => {
      const distances = session.output('distances', 'float32', pairCount);
      const initial = session.output('initial', 'float32', pairCount);
      const final = session.output('final', 'float32', pairCount);
      session.graph.add(
        new GPUGeodesicPairs({
          spatialContext: getSphereContext(),
          origins: session.input('origins', pack(from), 'float32x2'),
          targets: session.input('targets', pack(to), 'float32x2'),
          output: {
            distances: distances.view,
            initialBearings: initial.view,
            finalBearings: final.view
          }
        })
      );
      session.run();
      return {
        distances: await session.readFloat32(distances),
        initial: await session.readFloat32(initial),
        final: await session.readFloat32(final)
      };
    }
  );
  const table = makeCatalogTable(
    'geodesic-pairs',
    'Point pairs',
    {
      pair_id: 'string',
      origin_index: 'int32',
      destination_index: 'int32',
      distance: 'float64',
      initial_bearing: 'float64',
      final_bearing: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const wrap = (degrees: number) => (degrees + 360) % 360;
  for (let index = 0; index < pairCount; index++)
    appendCatalogRow(table, {
      pair_id: `${index}`,
      origin_index: index,
      destination_index: index,
      distance: result.distances[index],
      initial_bearing: wrap(result.initial[index]),
      final_bearing: wrap(result.final[index]),
      geometry: makeCatalogGeometryText('LineString', [from[index], to[index]])
    });
  const diagnostics: CatalogDiagnostic[] = [];
  if (from.length !== to.length)
    addCatalogDiagnostic(
      diagnostics,
      'unpaired-points',
      'warning',
      'Origins and destinations are paired by position; surplus points on the longer side are ignored.',
      Math.abs(from.length - to.length)
    );
  return {
    tables: [table],
    diagnostics,
    summary: {pairCount},
    dispatchedContributors: ['GPUGeodesicPairs']
  };
}
