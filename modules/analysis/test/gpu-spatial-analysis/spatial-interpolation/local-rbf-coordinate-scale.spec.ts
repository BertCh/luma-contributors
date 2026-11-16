// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {createVectorView} from '../../utils/gpu-contributor-test-utils';
import {createPaddedQueryFixture} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const POSITIONS = [0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2, 2, 2];
const QUERIES = [0.2, 0.35, 0.8, 1.4, 2, 2];

for (const [coordinateScale, smoothing, expected] of [
  [1e-7, 1e-35, [1.4779307483599453, 1.463224316857751, 1.4899402268042223]],
  [1e-9, 1e-36, [1.6094444612785554, 2.284444451399812, 2.4444444381331145]],
  [1e-9, 0, [1.7257008069325792, 0.5572898380836833, 1]]
] as const) {
  for (const sampleSmoothing of [false, true]) {
    it(`matches quintic smoothing ${smoothing} at scale ${coordinateScale} across 31/32/33-row chunks (${sampleSmoothing ? 'sample' : 'scalar'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createPaddedQueryFixture(device);
      const {graph, input, output} = fixture;
      const scale = Math.fround(coordinateScale);
      let rowBase = 0;
      const queries = [31, 32, 33].map((length, chunkIndex) => {
        const points = Float32Array.from(
          {length: length * 2},
          (_, component) => QUERIES[(rowBase * 2 + component) % 6] * scale
        );
        rowBase += length;
        return input(`queries-${chunkIndex}`, 'float32x2', points);
      });
      const values = [31, 32, 33].map((length, chunkIndex) =>
        output(`values-${chunkIndex}`, 'float32x4', length)
      );
      const statuses = [31, 32, 33].map((length, chunkIndex) =>
        output(`status-${chunkIndex}`, 'uint32', length)
      );
      graph.add(
        new GPULocalRBF({
          positions: input(
            'positions',
            'float32x2',
            Float32Array.from(POSITIONS, value => value * scale)
          ),
          values: input('values', 'float32', new Float32Array([1, 2, 0, 3, -1, 5, 2, 4, 1])),
          queries: createVectorView('queries', 'float32x2', queries),
          neighborCount: 9,
          kernel: 'quintic',
          smoothing: sampleSmoothing ? 0 : smoothing,
          sampleSmoothing: sampleSmoothing
            ? input('smoothing', 'float32', new Float32Array(9).fill(smoothing))
            : undefined,
          output: {
            values: createVectorView('results', 'float32x4', values),
            status: createVectorView('statuses', 'uint32', statuses)
          }
        })
      );
      const compiled = graph.compile();
      try {
        const actual = await fixture.submitTwice(compiled, async () => ({
          values: await fixture.readRows(values),
          status: await fixture.readRows(statuses)
        }));
        expect(actual.status).toEqual(Array(96).fill(0));
        // Independent SciPy RBFInterpolator fixtures on float32-rounded coordinates, quintic degree 2.
        // At 1e-9 the unscaled reciprocal fifth power overflows even for runtime zero smoothing.
        for (let row = 0; row < 96; row++)
          expect(actual.values[row * 4]).toBeCloseTo(expected[row % 3], 4);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

for (const sampleSmoothing of [false, true]) {
  for (const smoothing of [0, 1e-36]) {
    it(`normalizes quintic smoothing ${smoothing} at tiny scales (${sampleSmoothing ? 'sample' : 'scalar'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createPaddedQueryFixture(device);
      const {graph, input, output} = fixture;
      // Every old reciprocal pow(scale, 5) exceeds float32, even if subnormals are retained.
      const scale = Math.fround(1e-9);
      const positions = Float32Array.from(POSITIONS, value => value * scale);
      const queries = Float32Array.from(QUERIES, value => value * scale);
      // At zero smoothing a quadratic is reproduced independently of the radial weights.
      const values =
        smoothing === 0
          ? Float32Array.from({length: 9}, (_, row) => {
              const horizontal = POSITIONS[row * 2];
              const vertical = POSITIONS[row * 2 + 1];
              return 1 + horizontal + 2 * vertical + horizontal * vertical + vertical * vertical;
            })
          : new Float32Array([1, 2, 0, 3, -1, 5, 2, 4, 1]);
      const results = output('results', 'float32x4', 3);
      const statuses = output('status', 'uint32', 3);
      graph.add(
        new GPULocalRBF({
          positions: input('positions', 'float32x2', positions),
          values: input('values', 'float32', values),
          queries: input('queries', 'float32x2', queries),
          neighborCount: 9,
          kernel: 'quintic',
          smoothing: sampleSmoothing ? 0 : smoothing,
          sampleSmoothing: sampleSmoothing
            ? input('smoothing', 'float32', new Float32Array(9).fill(smoothing))
            : undefined,
          output: {values: results, status: statuses}
        })
      );
      const compiled = graph.compile();
      try {
        const actual = await fixture.submitTwice(compiled, async () => ({
          values: await fixture.readRows([results]),
          status: await fixture.readRows([statuses])
        }));
        expect(actual.status).toEqual([0, 0, 0]);
        // SciPy RBFInterpolator, quintic/degree 2, on the float32-rounded points scaled back
        // to unit coordinates, with smoothing=float32(1e-36)/float32(1e-9)^5.
        const expected =
          smoothing === 0
            ? Array.from({length: 3}, (_, row) => {
                const horizontal = queries[row * 2] / scale;
                const vertical = queries[row * 2 + 1] / scale;
                return 1 + horizontal + 2 * vertical + horizontal * vertical + vertical * vertical;
              })
            : [1.6094444612785554, 2.284444451399812, 2.4444444381331145];
        for (const [row, value] of expected.entries())
          expect(actual.values[row * 4]).toBeCloseTo(value, 4);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}
