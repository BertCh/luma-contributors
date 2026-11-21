// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {addWeightedQuantileLISA} from '../src/analysis/weighted-quantile-lisa';

test.each([
  ['greater', false],
  ['two-sided', false],
  ['geoda-legacy', false],
  ['greater', true],
  ['two-sided', true],
  ['geoda-legacy', true]
] as const)('constant cycle identifies %s with chunked=%s at both permutation budgets', async (alternative, chunked) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const permutationCount of [39, 99]) {
    const session = new CatalogGpuSession(
      device,
      `quantile-tail-${alternative}-${permutationCount}`
    );
    let result: ReturnType<typeof addWeightedQuantileLISA> | undefined;
    try {
      const source = {datasetId: 'constant-four-node-cycle', revision: 1};
      const mask = session.input('mask', Uint32Array.of(1, 1, 1, 1), 'uint32');
      const pValues = session.output('p-values', 'float32', 4);
      const overflow = session.output('overflow', 'uint32', 1);
      result = addWeightedQuantileLISA(session.graph, {
        source,
        population: 'all',
        values: session.input('values', Float32Array.of(4, 4, 4, 4), 'float32'),
        mask,
        settings: {
          quantileCount: 2,
          selectedQuantile: 2,
          seed: 7,
          permutationCount,
          significanceLevel: 0.05,
          alternative,
          multipleTesting: 'none',
          islands: 'exclude',
          quantileMethod: 'type-5'
        },
        topology: {
          source,
          mask,
          maximumNeighbors: 2,
          weights: chunked
            ? {
                rowCount: 4,
                information: {
                  rowCount: 4,
                  transform: 'binary',
                  symmetric: true,
                  hasSelfWeights: true
                },
                chunks: [0, 2].map(rowStart => ({
                  rowStart,
                  rowCount: 2,
                  weights: {
                    offsets: session.input(
                      `offsets-${rowStart}`,
                      Uint32Array.of(0, 3, 6),
                      'uint32'
                    ),
                    neighbors: session.input(
                      `neighbors-${rowStart}`,
                      rowStart === 0
                        ? Uint32Array.of(0, 1, 3, 1, 0, 2)
                        : Uint32Array.of(2, 1, 3, 3, 0, 2),
                      'uint32'
                    ),
                    weights: session.input(
                      `weights-${rowStart}`,
                      new Float32Array(6).fill(1),
                      'float32'
                    )
                  }
                }))
              }
            : {
                offsets: session.input('offsets', Uint32Array.of(0, 2, 4, 6, 8), 'uint32'),
                neighbors: session.input(
                  'neighbors',
                  Uint32Array.of(1, 3, 0, 2, 1, 3, 0, 2),
                  'uint32'
                ),
                weights: session.input('weights', new Float32Array(8).fill(1), 'float32'),
                information: {
                  rowCount: 4,
                  transform: 'binary',
                  symmetric: true,
                  hasSelfWeights: false
                }
              },
          status: {
            overflow: session.input('weights-overflow', Uint32Array.of(0), 'uint32'),
            requiredCount: session.input(
              'weights-required',
              Uint32Array.of(chunked ? 12 : 8),
              'uint32'
            ),
            invalidCount: session.input('weights-invalid', Uint32Array.of(0), 'uint32')
          }
        },
        outputs: {pseudoPValues: pValues.view, status: {overflow: overflow.view}}
      });
      expect(result.inference.alternative).toBe(alternative);
      session.run();
      const values = await session.readFloat32(pValues);
      expect(values).toHaveLength(4);
      expect((await session.readUint32(overflow))[0]).toBe(0);
      // Every ordered conditional draw has exactly 2 selected neighbors: all draws tie.
      // Inclusive alternatives give 1; legacy min(P,P-P) gives 1/(P+1).
      const expected =
        alternative === 'geoda-legacy' ? (permutationCount === 39 ? 0.025 : 0.01) : 1;
      for (const value of values) {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeCloseTo(expected, 6);
      }
    } finally {
      await session.dispose();
      result?.destroyParameters();
    }
  }
});

test('non-tied conditional joins distinguish inclusive greater from two-sided inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results: number[] = [];
  for (const alternative of ['greater', 'two-sided'] as const) {
    const session = new CatalogGpuSession(device, `non-tied-${alternative}`);
    let result: ReturnType<typeof addWeightedQuantileLISA> | undefined;
    try {
      const source = {datasetId: 'two-selected-members', revision: 1};
      const mask = session.input('mask', Uint32Array.of(1, 1, 1, 1), 'uint32');
      const pValues = session.output('p-values', 'float32', 4);
      result = addWeightedQuantileLISA(session.graph, {
        source,
        population: 'all',
        mask,
        values: session.input('values', Float32Array.of(9, 9, 1, 1), 'float32'),
        settings: {
          quantileCount: 2,
          selectedQuantile: 2,
          seed: 7,
          permutationCount: 999,
          significanceLevel: 0.05,
          alternative,
          multipleTesting: 'none',
          islands: 'exclude',
          quantileMethod: 'type-5'
        },
        topology: {
          source,
          mask,
          maximumNeighbors: 1,
          weights: {
            offsets: session.input('offsets', Uint32Array.of(0, 1, 2, 3, 4), 'uint32'),
            neighbors: session.input('neighbors', Uint32Array.of(1, 0, 3, 2), 'uint32'),
            weights: session.input('weights', new Float32Array(4).fill(1), 'float32'),
            information: {rowCount: 4, transform: 'binary', symmetric: true, hasSelfWeights: false}
          },
          status: {
            overflow: session.input('overflow', Uint32Array.of(0), 'uint32'),
            requiredCount: session.input('required', Uint32Array.of(4), 'uint32'),
            invalidCount: session.input('invalid', Uint32Array.of(0), 'uint32')
          }
        },
        outputs: {pseudoPValues: pValues.view}
      });
      session.run();
      const values = await session.readFloat32(pValues);
      expect(values).toHaveLength(4);
      // Focus 0 has one selected neighbor. Of the other three rows only row 1 is selected.
      // Exact conditional upper probability=1/3; inclusive lower=1; two-sided=2/3.
      // The 999 seeded Monte Carlo draws have 0.08 absolute tolerance around those literal
      // population expectations; using one finalizer for both alternatives cannot pass.
      const expected = alternative === 'greater' ? 1 / 3 : 2 / 3;
      for (const value of values.slice(0, 2)) {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - expected)).toBeLessThan(0.08);
      }
      results.push(values[0]);
    } finally {
      await session.dispose();
      result?.destroyParameters();
    }
  }
  expect(results[1] - results[0]).toBeGreaterThan(0.2);
});
