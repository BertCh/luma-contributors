// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {test} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {getCatalogBinding, runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';

function makeSquare(minimum: number, maximum: number): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: [
      {
        properties: {},
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [minimum, minimum],
              [maximum, minimum],
              [maximum, maximum],
              [minimum, maximum],
              [minimum, minimum]
            ]
          ]
        }
      }
    ]
  };
}

// Closed-form area oracles for the five newly styled GPU-only outputs. The public catalog
// execution path must actually dispatch GPU contributors; schema-only evidence is insufficient.
for (const [bindingId, expectedArea] of [
  ['geometry-line-buffer', 20 + Math.PI],
  ['geometry-polygon-buffer', 140 + Math.PI],
  ['geometry-overlay-intersection', 25],
  ['geometry-overlay-union', 175],
  ['geometry-overlay-difference', 75]
] as const) {
  test(`${bindingId}: GPU output area agrees with the presentation and legend`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const binding = getCatalogBinding(bindingId)!;
    const descriptor = getKeplerOperationDescriptor(bindingId)!;
    const result = await runCatalogOperationAsync(
      binding,
      Object.fromEntries(
        Object.entries({coordinateSystem: 'planar', distance: 1, quadSegments: 16}).filter(
          ([name]) => binding.parameters.some(parameter => parameter.name === name)
        )
      ),
      {
        lines: {
          type: 'FeatureCollection',
          features: [
            {
              properties: {},
              geometry: {
                type: 'LineString',
                coordinates: [
                  [0, 0],
                  [10, 0]
                ]
              }
            }
          ]
        },
        polygons: makeSquare(0, 10),
        left: makeSquare(0, 10),
        right: makeSquare(5, 15)
      },
      {device}
    );
    const output = descriptor.outputs[0];
    const table = result.tables.find(candidate => candidate.id === output.tableId);
    const plan = table && planKeplerResultPresentation(table, output);
    const values = table?.columns[output.presentation!.column];
    const totalArea = values?.reduce<number>(
      (sum, value) => sum + (typeof value === 'number' ? value : NaN),
      0
    );
    if (
      result.engine !== 'gpu' ||
      !result.dispatchedContributors.length ||
      !table?.rowCount ||
      !values?.every(value => typeof value === 'number' && Number.isFinite(value)) ||
      !plan ||
      plan.column !== 'area' ||
      !plan.ranges.length ||
      plan.ranges.reduce((total, range) => total + range.count, 0) !== table.rowCount ||
      totalArea === undefined ||
      Math.abs(totalArea / expectedArea - 1) > 0.01
    ) {
      throw new Error(
        `Output metadata mismatch: ${JSON.stringify({bindingId, engine: result.engine, contributors: result.dispatchedContributors, output, rowCount: table?.rowCount, columns: table && Object.keys(table.columns), values, expectedArea, totalArea, plan})}`
      );
    }
  });
}
