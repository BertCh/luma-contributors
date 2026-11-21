// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

type Fixture = 'perturbed' | 'tiny-C' | 'uniform-5' | 'uniform-8' | 'underflow';

const FIXTURES: {fixture: Fixture; scale: number}[] = [
  {fixture: 'perturbed', scale: 1},
  {fixture: 'perturbed', scale: 2 ** -60},
  {fixture: 'tiny-C', scale: 1},
  {fixture: 'uniform-5', scale: 1},
  {fixture: 'uniform-8', scale: 1},
  {fixture: 'underflow', scale: 1}
];

function getRowCount(fixture: Fixture): number {
  return {perturbed: 9, 'tiny-C': 4, 'uniform-5': 5, 'uniform-8': 8, underflow: 4}[fixture];
}

function getLabels(fixture: Fixture): Float32Array {
  const rows = getRowCount(fixture);
  if (fixture === 'perturbed') return Float32Array.from({length: rows}, (_, row) => row);
  if (fixture === 'tiny-C') return Float32Array.of(0, 0, 1, 1);
  if (fixture === 'underflow') return Float32Array.of(1, 1, 1, 1 + 2 ** -23);
  return Float32Array.from({length: rows}, (_, row) => Number(row === rows - 1));
}

function getWeight(fixture: Fixture, scale: number, row: number, neighbor: number): number {
  if (fixture === 'tiny-C') return row < 2 && neighbor < 2 ? 1 : 2 ** -60;
  if (fixture === 'perturbed') return scale * (row === 0 && neighbor === 1 ? 1 + 2 ** -23 : 1);
  // Review B3: complete weights 2^-112 with w01 doubled; the preliminary float32 C is 0/0.
  if (fixture === 'underflow') return (row === 0 && neighbor === 1 ? 2 : 1) * 2 ** -112;
  return 1;
}

for (const mode of ['reference', 'tiled'] as const)
  for (const chunked of [false, true])
    it(`W5-17 Geary preserves tiny variance and positive C (${mode}, ${chunked ? 'chunked' : 'whole'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const {fixture, scale} of FIXTURES) {
        const rig = new WeightsRig(device);
        const rows = getRowCount(fixture);
        const upload = (start: number, count: number) => {
          const neighbors = Array.from({length: count}, (_, local) =>
            Array.from({length: rows}, (_, neighbor) => neighbor).filter(
              neighbor => neighbor !== start + local
            )
          );
          return rig.uploadWeights({
            offsets: Array.from({length: count + 1}, (_, row) => row * (rows - 1)),
            neighbors: neighbors.flat(),
            weights: neighbors.flatMap((entries, local) =>
              entries.map(neighbor => getWeight(fixture, scale, start + local, neighbor))
            ),
            distances: []
          });
        };
        try {
          const summary = rig.output('float32', 52);
          rig.run(
            new GPUGlobalSpatialStatistics({
              mode,
              statistics: ['geary'],
              values: rig.input(getLabels(fixture), 'float32', rows),
              weights: chunked
                ? {
                    rowCount: rows,
                    chunks: [
                      {rowStart: 0, rowCount: 2, weights: upload(0, 2)},
                      {rowStart: 2, rowCount: rows - 2, weights: upload(2, rows - 2)}
                    ]
                  }
                : upload(0, rows),
              output: {summary: summary.view}
            })
          );
          const result = await readFloat32(summary.buffer, 52);
          const label = `${fixture} scale ${scale}`;
          if (fixture === 'uniform-5' || fixture === 'uniform-8') {
            // Review B2: every permutation has Q = 2*n*M2 and S0 = n*(n-1), so C = 1 exactly
            // (the direct quadratic sum rounds to 1 + 2^-23 for eight rows in float32).
            expect(result[16], label).toBe(1);
            continue;
          }
          if (fixture === 'underflow') {
            // Exact Fraction enumeration of all 4! permutations
            // (.typecheck/r19/scratch/spatial-stats-w5-local-global-and-rate-statistics/geary-b3-reference.py):
            // C = 12/13, randomization variance 1/169, z = -1.
            for (const [field, expected] of [
              [16, 12 / 13],
              [21, 1 / 169],
              [22, -1]
            ] as const) {
              expect(Number.isFinite(result[field]), `${label} field ${field}`).toBe(true);
              expect(
                Math.abs(result[field] / expected - 1),
                `${label} field ${field}`
              ).toBeLessThan(2e-4);
            }
            continue;
          }
          const field = fixture === 'perturbed' ? 21 : 16;
          // Independent Fraction endpoint enumeration and direct quadratic derivation are
          // recorded in geary-small-weights.node.spec.ts; these tolerances explicitly reject 0.
          const expected = fixture === 'perturbed' ? 3.0702463789201337e-18 : 5.204170427930421e-18;
          expect(Number.isFinite(result[field]), label).toBe(true);
          expect(result[field], label).toBeGreaterThan(0);
          expect(Math.abs(result[field] / expected - 1), label).toBeLessThan(2e-4);
          if (fixture === 'perturbed') {
            // Near-null inference must keep the centered C-1 calculation.
            expect(Number.isFinite(result[22]), label).toBe(true);
            expect(Math.abs(result[22] / -0.8819171037 - 1), label).toBeLessThan(2e-4);
            expect(Number.isFinite(result[18]), label).toBe(true);
            expect(result[18], label).toBeGreaterThan(0);
            expect(Math.abs(result[18] / 3.8378079736501665e-18 - 1), label).toBeLessThan(2e-4);
          }
        } finally {
          rig.destroy();
        }
      }
    });
