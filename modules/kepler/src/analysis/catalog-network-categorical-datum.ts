// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUDatumGridTransform} from '@luma-contributors/analysis/geospatial-analysis';
import type {CatalogOperationBinding} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {
  getPointPositions,
  makeCatalogGeometryText,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {
  makeInputSpec,
  makeOutputSpec,
  makeNumberParameter,
  makeStringParameter
} from './catalog-platform3-b-common';
import {
  getNumber,
  getString,
  getSourceRow,
  makeBinding,
  makeEnum,
  makeResult,
  readPositions
} from './catalog-network-categorical-common';

export function prepareDatumSamples(
  samples: CatalogFeatureCollection,
  width: number,
  height: number
) {
  if (width * height !== samples.features.length)
    throw new Error('Datum grid width * height must equal the sample row count (row-major order)');
  const mask = Uint32Array.from(
    samples.features.map(feature =>
      feature.properties?.['valid'] === false || feature.properties?.['valid'] === 0 ? 0 : 1
    )
  );
  const shifts = Float32Array.from(
    samples.features.flatMap((feature, index) => {
      if (!mask[index]) return [0, 0, 0];
      return ['longitudeShift', 'latitudeShift', 'heightShift'].map(property => {
        const value = feature.properties?.[property];
        if (typeof value !== 'number' || !Number.isFinite(Math.fround(value)))
          throw new Error(`Datum grid ${property} must be finite on valid samples`);
        return value;
      });
    })
  );
  return {mask, shifts};
}

/** Pack exactly three high/residual words, preserving geographic altitude or a named height. */
export function prepareDatumPositions(
  points: CatalogFeatureCollection,
  heightProperty: string,
  requireHeight: boolean
) {
  readPositions(points, true);
  const original = points.features.flatMap(feature => {
    const position = getPointPositions(feature.geometry)[0];
    const height =
      feature.properties?.[heightProperty] ??
      (position as readonly number[])[2] ??
      (requireHeight ? undefined : 0);
    if (typeof height !== 'number' || !Number.isFinite(Math.fround(height)))
      throw new Error('Input height must be finite');
    return [position[0], position[1], height];
  });
  const high = Float32Array.from(original);
  const low = Float32Array.from(original.map((value, index) => value - high[index]));
  return {high, low};
}

export const DATUM_BINDING: CatalogOperationBinding = makeBinding({
  id: 'datum-grid-transform',
  title: 'Datum grid transform',
  domain: 'geometry',
  summary:
    'Apply a caller-decoded regular datum shift grid with bilinear interpolation and bounded inverse iteration.',
  contributorNames: ['GPUDatumGridTransform'],
  inputs: [
    makeInputSpec('points', 'Longitude/latitude points', ['point'], 20000, 20000),
    makeInputSpec('samples', 'Decoded datum grid samples (row-major)', ['point'], 65536, 65536)
  ],
  parameters: [
    makeNumberParameter('width', 'Grid width', 2, 2, 256, undefined, 'integer'),
    makeNumberParameter('height', 'Grid height', 2, 2, 256, undefined, 'integer'),
    makeNumberParameter('originLongitude', 'Grid origin longitude', -180, -360, 360, 'degrees'),
    makeNumberParameter('originLatitude', 'Grid origin latitude', -90, -90, 90, 'degrees'),
    makeNumberParameter('spacingLongitude', 'Longitude spacing', 360, 1e-8, 720, 'degrees'),
    makeNumberParameter('spacingLatitude', 'Latitude spacing', 180, 1e-8, 180, 'degrees'),
    makeEnum('operation', 'Datum operation', ['horizontal', 'vertical', 'combined']),
    makeEnum('direction', 'Datum direction', ['forward', 'inverse']),
    makeStringParameter('heightProperty', 'Height property (or point altitude)', 'height'),
    makeStringParameter('sourceVerticalReference', 'Source vertical reference', 'source'),
    makeStringParameter('targetVerticalReference', 'Target vertical reference', 'target'),
    makeNumberParameter('heightUnitMeters', 'Meters per height unit', 1, 1e-8, 1e8),
    makeNumberParameter(
      'maximumIterations',
      'Inverse iteration limit',
      10,
      1,
      64,
      undefined,
      'integer'
    ),
    makeNumberParameter('tolerance', 'Inverse residual tolerance', 5e-9, 1e-12, 1, 'degrees')
  ],
  outputs: [
    makeOutputSpec(
      'datum-grid-transform',
      'Datum transformed points',
      'point',
      ['source_index', 'source_id'],
      20000
    )
  ],
  declaredBehavior: [
    'One decoded regular grid; sample row order defines the grid, and sample geometry is not interpreted. Each sample supplies longitudeShift/latitudeShift in degrees and heightShift in meters. valid=false or 0 marks nodata.',
    'All four interpolation corners must be valid. Input longitude must use the same continuous branch as the grid, including at the antimeridian.',
    'Input coordinates use Float32 high and residual words; output coordinates recombine them. Status bits are retained: 0 success, 1 invalid, 2 no coverage, 4 nodata, 8 inverse exhausted. Failed rows retain source ids and null output geometry.',
    'A height property overrides point altitude. Horizontal operations preserve supplied heights (default zero when both are absent). Vertical/combined operations require finite input heights and explicit source/target vertical references. Inverse exhaustion retains its status and does not publish a successful geometry.'
  ],
  limitation:
    'Caller-decoded samples only; no NTv2/GeoTIFF download or parser, multiple grid precedence or CRS reprojection. Grid spacing is positive.',
  runGpu: async (context, inputs, values) => {
    const points = inputs['points'];
    const operation = values['operation'] as 'horizontal' | 'vertical' | 'combined';
    const {high, low} = prepareDatumPositions(
      points,
      getString(values, 'heightProperty'),
      operation !== 'horizontal'
    );
    const width = getNumber(values, 'width');
    const height = getNumber(values, 'height');
    const samples = prepareDatumSamples(inputs['samples'], width, height);
    return withCatalogGpuSession(context.device, 'catalog-datum-grid', async session => {
      const positions = session.output('positions', 'float32x3', points.features.length);
      const positionsLow = session.output('positions-low', 'float32x3', points.features.length);
      const status = session.output('status', 'uint32', points.features.length);
      session.graph.add(
        new GPUDatumGridTransform({
          positions: session.input('input', high, 'float32x3'),
          positionsLow: session.input('input-low', low, 'float32x3'),
          samples: session.input('samples', samples.shifts, 'float32x3'),
          mask: session.input('mask', samples.mask, 'uint32'),
          grids: [
            {
              width,
              height,
              sampleOffset: 0,
              origin: [getNumber(values, 'originLongitude'), getNumber(values, 'originLatitude')],
              spacing: [getNumber(values, 'spacingLongitude'), getNumber(values, 'spacingLatitude')]
            }
          ],
          operation,
          direction: values['direction'] as 'forward' | 'inverse',
          sourceVerticalReference: getString(values, 'sourceVerticalReference'),
          targetVerticalReference: getString(values, 'targetVerticalReference'),
          heightUnitMeters: getNumber(values, 'heightUnitMeters'),
          maximumIterations: getNumber(values, 'maximumIterations'),
          tolerance: getNumber(values, 'tolerance'),
          output: {positions: positions.view, positionsLow: positionsLow.view, status: status.view}
        })
      );
      session.run();
      const output = await session.readFloat32(positions);
      const residuals = await session.readFloat32(positionsLow);
      const statuses = await session.readUint32(status);
      return makeResult(
        'datum-grid-transform',
        'Datum transformed points',
        {
          source_index: 'int32',
          source_id: 'string',
          geometry: 'string',
          longitude: 'float64',
          latitude: 'float64',
          height: 'float64',
          status: 'int32'
        },
        points.features.map((_, index) => {
          const coordinates = [0, 1, 2].map(
            component => output[index * 3 + component] + residuals[index * 3 + component]
          );
          return {
            ...getSourceRow(points, index),
            longitude: coordinates[0],
            latitude: coordinates[1],
            height: coordinates[2],
            status: statuses[index],
            geometry: statuses[index] === 0 ? makeCatalogGeometryText('Point', coordinates) : null
          };
        }),
        {
          sourceVerticalReference: getString(values, 'sourceVerticalReference'),
          targetVerticalReference: getString(values, 'targetVerticalReference'),
          failedCount: statuses.filter(status => status !== 0).length
        }
      );
    });
  }
});
