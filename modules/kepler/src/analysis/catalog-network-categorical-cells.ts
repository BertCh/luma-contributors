// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUHealpixPointToCell,
  GPUHealpixCellAggregation,
  getHealpixCellBoundary
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogOperationBinding} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {makeCatalogGeometryText} from './catalog-geojson';
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
  makeResult,
  readPositions,
  readNumbers
} from './catalog-network-categorical-common';

export const CELL_BINDINGS: readonly CatalogOperationBinding[] = (
  ['healpix-point-to-cell', 'healpix-cell-aggregation'] as const
).map(id => {
  const aggregation = id === 'healpix-cell-aggregation';
  const title = aggregation ? 'HEALPix cell aggregation' : 'HEALPix point to cell';
  return makeBinding({
    id,
    title,
    domain: 'aggregation',
    summary: aggregation
      ? 'Aggregate geographic points into equal-area spherical HEALPix cells and publish cell polygons.'
      : 'Assign spherical nested HEALPix cell ids to source points.',
    contributorNames: [
      'GPUHealpixPointToCell',
      ...(aggregation ? ['GPUHealpixCellAggregation'] : [])
    ],
    inputs: [makeInputSpec('points', 'Longitude/latitude points', ['point'], 20000, 20000)],
    parameters: [
      makeNumberParameter('order', 'HEALPix order', 8, 0, 14, undefined, 'integer'),
      ...(aggregation
        ? [
            makeStringParameter('valueProperty', 'Value property (optional)', ''),
            makeNumberParameter(
              'cellCapacity',
              'Cell output capacity',
              20000,
              1,
              20000,
              undefined,
              'integer'
            )
          ]
        : [])
    ],
    outputs: [
      makeOutputSpec(
        id,
        title,
        aggregation ? 'polygon' : 'point',
        aggregation ? ['cell_id'] : ['source_index', 'source_id'],
        20000
      )
    ],
    declaredBehavior: [
      'One finite longitude/latitude point per source feature; latitude must be in [-90, 90]. Coordinates are rounded to Float32, and cell-edge assignments can differ from Float64.',
      'Nested keys are published as exact decimal strings, including valid cell zero; validity is passed to the aggregation mask.',
      'Cell counts, sums, minima and maxima come from GPUHealpixCellAggregation. Missing/invalid optional values fail before dispatch. Capacity overflow is reported and prevents partial publication.',
      'Polygons sample each curved spherical edge with eight segments. Longitudes are unwrapped across the antimeridian; no ellipsoidal authalic conversion is performed.'
    ],
    limitation:
      'Nested scheme, orders 0-14 (higher orders suffer geographic Float32 quantization). Spherical cell geometry is sampled display geometry; not OGC ZIRS.',
    runGpu: async (context, inputs, values) => {
      const points = inputs['points'];
      const positions = readPositions(points, true);
      const order = getNumber(values, 'order');
      const property = aggregation ? getString(values, 'valueProperty') : '';
      const numericValues = property ? readNumbers(points, property) : undefined;
      return withCatalogGpuSession(context.device, `catalog-${id}`, async session => {
        const cells = session.output('cells', 'uint32x2', points.features.length);
        const validity = session.output('validity', 'uint32', points.features.length);
        session.graph.add(
          new GPUHealpixPointToCell({
            order,
            positions: session.input('positions', positions, 'float32x2'),
            output: {cells: cells.view, validity: validity.view}
          })
        );
        const capacity = aggregation ? getNumber(values, 'cellCapacity') : points.features.length;
        const aggregateCells = session.output('aggregate-cells', 'uint32x2', capacity);
        const counts = session.output('counts', 'uint32', capacity);
        const sums = session.output('sums', 'float32', capacity);
        const minimums = session.output('minimums', 'float32', capacity);
        const maximums = session.output('maximums', 'float32', capacity);
        const count = session.output('count', 'uint32', 1);
        const required = session.output('required', 'uint32', 1);
        const overflow = session.output('overflow', 'uint32', 1);
        const invalid = session.output('invalid', 'uint32', 1);
        if (aggregation)
          session.graph.add(
            new GPUHealpixCellAggregation({
              order,
              cells: cells.view,
              mask: validity.view,
              values: numericValues ? session.input('values', numericValues, 'float32') : undefined,
              cellCapacity: capacity,
              output: {
                cells: aggregateCells.view,
                counts: counts.view,
                ...(numericValues
                  ? {sums: sums.view, minimums: minimums.view, maximums: maximums.view}
                  : {}),
                status: {
                  count: count.view,
                  requiredCount: required.view,
                  overflow: overflow.view,
                  invalidCount: invalid.view
                }
              }
            })
          );
        session.run();
        const words = await session.readUint32(aggregation ? aggregateCells : cells);
        const key = (index: number) =>
          BigInt(words[index * 2]) | (BigInt(words[index * 2 + 1]) << 32n);
        if (!aggregation) {
          const valid = await session.readUint32(validity);
          return makeResult(
            id,
            title,
            {
              source_index: 'int32',
              source_id: 'string',
              geometry: 'string',
              cell_id: 'string',
              order: 'int32',
              validity: 'boolean'
            },
            points.features.map((_, index) => ({
              ...getSourceRow(points, index),
              cell_id: valid[index] ? key(index).toString() : null,
              order,
              validity: valid[index] !== 0
            }))
          );
        }
        const requiredCount = await session.readScalar(required);
        if (await session.readScalar(overflow))
          throw new Error(
            `HEALPix cell output capacity ${capacity} exceeded; requiredCount=${requiredCount}. Increase cell output capacity.`
          );
        const rowCount = await session.readScalar(count);
        const cellCounts = await session.readUint32(counts);
        const cellSums = numericValues ? await session.readFloat32(sums) : undefined;
        const cellMinimums = numericValues ? await session.readFloat32(minimums) : undefined;
        const cellMaximums = numericValues ? await session.readFloat32(maximums) : undefined;
        return makeResult(
          id,
          title,
          {
            cell_id: 'string',
            order: 'int32',
            count: 'int32',
            sum: 'float64',
            minimum: 'float64',
            maximum: 'float64',
            geometry: 'string'
          },
          Array.from({length: rowCount}, (_, index) => ({
            cell_id: key(index).toString(),
            order,
            count: cellCounts[index],
            sum: cellSums?.[index] ?? null,
            minimum: cellMinimums?.[index] ?? null,
            maximum: cellMaximums?.[index] ?? null,
            geometry: makeCatalogGeometryText('Polygon', [
              getHealpixCellBoundary(key(index), order)
            ])
          })),
          {requiredCount, overflow: 0, invalidCount: await session.readScalar(invalid)}
        );
      });
    }
  });
});
