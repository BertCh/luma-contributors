// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const [scale, incomplete] of [
  [1, false],
  [3, false],
  [1, true]
] as const) {
  it(`audits all partition-evaluation outputs against NumPy/sklearn at scale ${scale}, incomplete ${incomplete}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const shift = scale === 3 ? 64 : 0;
      const values = Float32Array.of(
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        1,
        10,
        3,
        12,
        5,
        20,
        9,
        22,
        11,
        100,
        100,
        NaN,
        0,
        0,
        Infinity,
        -100,
        -100
      ).map(value => value * scale + shift);
      const summary = rig.output('float32', 9);
      const regionSizes = rig.output('uint32', 5);
      const regionMeans = rig.output('float32', 10);
      const regionWithin = rig.output('float32', 5);
      const invalidRowCount = rig.output('uint32', 1);
      const neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
      rig.graph.add(
        new GPURegionPartitionEvaluation({
          values: rig.input(values, 'float32', 24),
          labels: rig.input(
            Uint32Array.of(0, 0, 0, 2, 2, 2, 3, 3, 4294967295, 0, 2, 9),
            'uint32',
            12
          ),
          columnCount: 2,
          labelCapacity: 5,
          weights: {
            ...rig.uploadWeights({
              offsets: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11],
              neighbors,
              weights: [0, -1, 2, 3, 100, 0, 1, 2, 3, 4, 5],
              distances: []
            }),
            status: {candidateOverflow: rig.input(Uint32Array.of(Number(incomplete)), 'uint32', 1)}
          },
          output: {
            summary: summary.view,
            regionSizes: regionSizes.view,
            regionMeans: regionMeans.view,
            regionWithinSumOfSquares: regionWithin.view,
            invalidRowCount: invalidRowCount.view
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        // Oracle: regionalization-audit-evaluation.oracle.py; complete-row exclusion, unweighted
        // CSR slot fraction 2/7, no variance divisor, and sklearn Calinski-Harabasz divisors.
        const expectedSummary =
          scale === 1
            ? [3, 36, 507, 543, 2, 3, 2 / 7, 4, 35.208333333333336]
            : [3, 324, 4563, 4887, 2, 3, 2 / 7, 4, 35.208333333333336];
        if (incomplete) expectedSummary[6] = NaN;
        const expectedMeans =
          scale === 1
            ? [2, 3, NaN, NaN, 10, 3, 21, 10, NaN, NaN]
            : [70, 73, NaN, NaN, 94, 73, 127, 94, NaN, NaN];
        const compare = (actual: number[], expected: number[]) => {
          expect(actual).toHaveLength(expected.length);
          expected.forEach((value, index) => {
            if (Number.isNaN(value)) expect(Number.isNaN(actual[index])).toBe(true);
            else {
              expect(Number.isFinite(actual[index])).toBe(true);
              // Fixed small inputs, float32 division and summation: 1e-5 relative/absolute.
              expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(
                1e-5 * Math.max(1, Math.abs(value))
              );
            }
          });
        };
        compare(await readFloat32(summary.buffer, 9), expectedSummary);
        compare(await readFloat32(regionMeans.buffer, 10), expectedMeans);
        compare(
          await readFloat32(regionWithin.buffer, 5),
          [16, 0, 16, 4, 0].map(value => value * scale ** 2)
        );
        expect(await readUint32(regionSizes.buffer, 5)).toEqual([3, 0, 3, 2, 0]);
        expect(await readUint32(invalidRowCount.buffer, 1)).toEqual([2]);
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });
}

it('v2-1 reduces every row and label tile, including ignored and invalid tail rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const summary = rig.output('float32', 9);
    const regionSizes = rig.output('uint32', 260);
    const regionMeans = rig.output('float32', 260);
    const regionWithin = rig.output('float32', 260);
    const invalid = rig.output('uint32', 1);
    rig.graph.add(
      new GPURegionPartitionEvaluation({
        values: rig.input(
          Float32Array.from([...Array.from({length: 256}, (_, row) => row % 2), 10, 12, NaN, 99]),
          'float32',
          260
        ),
        labels: rig.input(
          Uint32Array.from([...new Array(256).fill(0), 259, 259, 0, 4294967295]),
          'uint32',
          260
        ),
        labelCapacity: 260,
        weights: rig.uploadWeights({
          offsets: [0, ...Array.from({length: 255}, () => 1), 1, 2, 3, 4, 5],
          neighbors: [256, 0, 256, 0, 0],
          weights: [1, 1, 1, 1, 1],
          distances: []
        }),
        output: {
          summary: summary.view,
          regionSizes: regionSizes.view,
          regionMeans: regionMeans.view,
          regionWithinSumOfSquares: regionWithin.view,
          invalidRowCount: invalid.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // NumPy/sklearn oracle in the companion .oracle.py. No variance divisor in SSD.
      const expected = [
        2,
        66,
        218.7906976744186,
        284.7906976744186,
        2,
        256,
        2 / 3,
        2,
        848.6427061310782
      ];
      (await readFloat32(summary.buffer, 9)).forEach((value, index) => {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - expected[index])).toBeLessThanOrEqual(
          1e-5 * Math.max(1, Math.abs(expected[index]))
        );
      });
      expect(await readUint32(invalid.buffer, 1)).toEqual([1]);
      expect(await readUint32(regionSizes.buffer, 260)).toEqual([
        256,
        ...new Array(258).fill(0),
        2
      ]);
      expect(await readFloat32(regionWithin.buffer, 260)).toEqual([
        64,
        ...new Array(258).fill(0),
        2
      ]);
      const means = await readFloat32(regionMeans.buffer, 260);
      expect(means[0]).toBe(0.5);
      expect(means[259]).toBe(11);
      means.slice(1, 259).forEach(value => expect(Number.isNaN(value)).toBe(true));
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});
