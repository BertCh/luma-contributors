// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {getGPURasterFractionalCoverageParameterValues} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
  it(`B2/B5 getter-backed loss and versioned area metadata (${cellSizeMode})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const candidateLoss of [0, 1]) {
      for (const layout of ['legacy', 'local'] as const) {
        const rig = createParityGraph(device);
        const originY = cellSizeMode === 'geographic' ? 25 : 4600000;
        const grid = {
          width: 1,
          height: 1,
          affine: [1, 0, 0, 0, 1, originY] as const,
          pixelInterpretation: 'area' as const
        };
        const parameters =
          layout === 'local'
            ? getGPURasterFractionalCoverageParameterValues(grid, {localOrigin: [0, originY]})
            : Float32Array.of(0, originY, 1, 0, 0, 1, 1, 31);
        const area = rig.output('area', 1, 'float32');
        const validity = rig.output('validity', 1, 'uint32');
        const loss = rig.input('candidate-loss', [candidateLoss], 'uint32').view;
        const status = Object.assign(
          Object.create({
            get candidateOverflow() {
              return loss;
            }
          }),
          {
            count: rig.input('count', [1], 'uint32').view,
            requiredCount: rig.input('required', [1], 'uint32').view,
            overflow: rig.input('overflow', [0], 'uint32').view
          }
        );
        rig.graph.add(
          new GPURasterFractionalCategoricalStatistics({
            grid,
            cellSizeMode,
            zoneCount: 1,
            categoryCapacity: 1,
            parameters: rig.input('parameters', Array.from(parameters), 'float32').view,
            input: {
              id: 'categories',
              format: 'uint32',
              storage: {kind: 'buffer', values: rig.input('categories', [0], 'uint32').view}
            },
            coverage: {
              zoneIds: rig.input('zones', [0], 'uint32').view,
              cellIds: rig.input('cells', [0], 'uint32').view,
              fractions: rig.input('fractions', [1], 'float32').view,
              zoneValidity: rig.input('zone-validity', [1], 'uint32').view,
              status
            },
            output: {
              coverageSums: rig.output('coverage-sums', 1, 'float32').view,
              classAreas: area.view,
              proportions: rig.output('proportions', 1, 'float32').view,
              coveredArea: rig.output('covered-area', 1, 'float32').view,
              validArea: rig.output('valid-area', 1, 'float32').view,
              missingArea: rig.output('missing-area', 1, 'float32').view,
              majority: rig.output('majority', 1, 'uint32').view,
              minority: rig.output('minority', 1, 'uint32').view,
              validity: validity.view
            }
          })
        );
        const compiled = rig.graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readUint32(validity.buffer, 1)).toEqual([1 - candidateLoss]);
          const [value] = await readFloat32(area.buffer, 1);
          if (candidateLoss) expect(Number.isNaN(value)).toBe(true);
          else {
            // Independent WGS84 midpoint N*M*cos(latitude)*(pi/180)^2 for degrees;
            // N*M*cos(latitude)^2/a^2 for Mercator. See coverage-local-origin fixture derivation.
            const expected = cellSizeMode === 'geographic' ? 11137607818.7 : 0.6175050465639351;
            expect(Number.isFinite(value)).toBe(true);
            expect(value / expected).toBeCloseTo(1, 5);
          }
        } finally {
          compiled.destroy();
          rig.destroy();
        }
      }
    }
  });
}
