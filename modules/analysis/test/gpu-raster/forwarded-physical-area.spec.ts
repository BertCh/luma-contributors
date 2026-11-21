// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {
  GPURasterFractionalZonalStatistics,
  getGPURasterFractionalCoverageParameterValues
} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalCategoricalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {createParityGraph} from '../oracles/parity-wp2-geodesic-harness';
import {readFloat32, submitGraph} from '../utils/gpu-contributor-test-utils';

for (const packed of [false, true])
  for (const valid of [0, 1]) {
    it(`B1 physical scalar and categorical areas packed=${packed} valid=${valid}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = createParityGraph(device);
      const grid = {
        width: 1,
        height: 1,
        affine: [2, 1, 0, 4, 2 + 5 * 2 ** -24, 0] as const,
        pixelInterpretation: 'area' as const
      };
      // Original affine determinant = 2*(2+5*2^-24)-4 = 5*2^-23.
      const area = 5.960464477539062e-7;
      const parameters = packed
        ? rig.input(
            'parameters',
            [...getGPURasterFractionalCoverageParameterValues(grid)],
            'float32'
          ).view
        : undefined;
      const coverage = {
        zoneIds: rig.input('zones', [0], 'uint32').view,
        cellIds: rig.input('cells', [0], 'uint32').view,
        fractions: rig.input('fractions', [1], 'float32').view,
        zoneValidity: rig.input('zone-validity', [1], 'uint32').view,
        status: {
          count: rig.input('count', [1], 'uint32').view,
          requiredCount: rig.input('required', [1], 'uint32').view,
          overflow: rig.input('overflow', [0], 'uint32').view
        }
      };
      const scalar = Object.fromEntries(
        ['totals', 'means', 'coveredArea', 'validArea'].map(name => [
          name,
          rig.output(`scalar-${name}`, 1, 'float32')
        ])
      );
      const category = Object.fromEntries(
        [
          'coverageSums',
          'classAreas',
          'proportions',
          'coveredArea',
          'validArea',
          'missingArea'
        ].map(name => [name, rig.output(`category-${name}`, 1, 'float32')])
      );
      rig.graph.add(
        new GPURasterFractionalZonalStatistics({
          id: 'scalar',
          grid,
          zoneCount: 1,
          parameters,
          quantity: 'density',
          coverage,
          input: {
            id: 'values',
            format: 'float32',
            storage: {kind: 'buffer', values: rig.input('values', [1], 'float32').view},
            validity: rig.input('valid', [valid], 'uint32').view
          },
          output: {
            totals: scalar['totals'].view,
            means: scalar['means'].view,
            coveredArea: scalar['coveredArea'].view,
            validArea: scalar['validArea'].view,
            validity: rig.output('scalar-validity', 1, 'uint32').view
          }
        })
      );
      rig.graph.add(
        new GPURasterFractionalCategoricalStatistics({
          id: 'category',
          grid,
          zoneCount: 1,
          categoryCapacity: 1,
          parameters,
          coverage,
          input: {
            id: 'categories',
            format: 'uint32',
            storage: {kind: 'buffer', values: rig.input('categories', [0], 'uint32').view},
            validity: rig.input('category-valid', [valid], 'uint32').view
          },
          output: {
            coverageSums: category['coverageSums'].view,
            classAreas: category['classAreas'].view,
            proportions: category['proportions'].view,
            coveredArea: category['coveredArea'].view,
            validArea: category['validArea'].view,
            missingArea: category['missingArea'].view,
            majority: rig.output('majority', 1, 'uint32').view,
            minority: rig.output('minority', 1, 'uint32').view,
            validity: rig.output('category-validity', 1, 'uint32').view
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const checks = [
          [scalar['coveredArea'], area],
          [scalar['validArea'], area * valid],
          [category['coveredArea'], area],
          [category['validArea'], area * valid],
          [category['classAreas'], area * valid],
          [category['missingArea'], area * (1 - valid)],
          ...(valid ? [[scalar['totals'], area] as const] : [])
        ] as const;
        for (const [output, expected] of checks) {
          const [received] = await readFloat32(output.buffer, 1);
          expect(Number.isFinite(received)).toBe(true);
          expect(Math.abs(received - expected)).toBeLessThan(area * 2e-6);
        }
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    });
  }
