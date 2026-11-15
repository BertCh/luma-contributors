// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCellTopology,
  getCellTopologyStride,
  type GPUCellTopologyOperation
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-topology';
import {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  getS2TopologyCell,
  getS2TopologyRowOnCPU,
  S2_TOPOLOGY_REFERENCE
} from './s2-topology-oracle';

const INVALID_KEYS = [0n, 2n, 0xc000000000000001n, 0xffffffffffffffffn];

async function checkRows(
  operation: GPUCellTopologyOperation,
  keys: bigint[],
  wordOrder: 'little-endian' | 'high-low',
  withMask: boolean
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const stride = getCellTopologyStride('s2', operation);
  const mask = Uint32Array.from(keys.map((_, row) => (row % 7 === 0 ? 0 : row % 3 === 0 ? 7 : 1)));
  const cells = Uint32Array.from(
    keys.flatMap(key => {
      const words = [Number(key & 0xffffffffn), Number(key >> 32n)];
      return wordOrder === 'high-low' ? words.reverse() : words;
    })
  );
  const result = await runOperationGraph(device, {
    inputs: {
      cells: {format: 'uint32x2', values: cells},
      ...(withMask ? {mask: {format: 'uint32' as const, values: mask}} : {})
    },
    operations: [
      new GPUContributorOperation({
        id: 's2-topology-test',
        inputs: {cells: 'cells', ...(withMask ? {mask: 'mask'} : {})},
        outputs: {
          outputCells: {value: 'outputCells', format: 'uint32x2', length: keys.length * stride},
          counts: {value: 'counts', format: 'uint32', length: keys.length},
          ...(operation.type === 'disk'
            ? {
                distances: {
                  value: 'distances',
                  format: 'uint32' as const,
                  length: keys.length * stride
                }
              }
            : {})
        },
        createContributor: getView =>
          new GPUCellTopology({
            family: 's2',
            operation,
            wordOrder,
            cells: getView<'uint32x2'>('cells'),
            mask: withMask ? getView<'uint32'>('mask') : undefined,
            output: {
              cells: getView<'uint32x2'>('outputCells'),
              counts: getView<'uint32'>('counts'),
              distances: operation.type === 'disk' ? getView<'uint32'>('distances') : undefined
            }
          })
      })
    ],
    outputs: ['outputCells', 'counts', ...(operation.type === 'disk' ? ['distances'] : [])]
  });
  keys.forEach((key, row) => {
    const expected =
      withMask && mask[row] === 0
        ? {cells: [], distances: []}
        : getS2TopologyRowOnCPU(operation, key);
    expect(result['counts'][row], `row ${row}, key ${key.toString(16)}`).toBe(
      expected.cells.length
    );
    for (let slot = 0; slot < stride; slot++) {
      const offset = row * stride + slot;
      const actual =
        BigInt(result['outputCells'][offset * 2]) |
        (BigInt(result['outputCells'][offset * 2 + 1]) << 32n);
      expect(actual, `row ${row}, slot ${slot}, key ${key.toString(16)}`).toBe(
        expected.cells[slot] ?? 0n
      );
      if (operation.type === 'disk')
        expect(result['distances'][offset], `row ${row}, slot ${slot} distance`).toBe(
          expected.distances[slot] ?? 0
        );
    }
  });
}

for (const type of ['disk', 'ring'] as const)
  for (const radius of [0, 1, 2, 8])
    it(`S2 ${type} radius ${radius} matches the pinned cube-face graph, including padding and masks`, async () => {
      const keys = S2_TOPOLOGY_REFERENCE.filter(row => radius <= 1 || row.level <= 2).map(
        row => row.cell
      );
      // Interior neighborhoods attain the flat-grid stride at fine levels. Neighbors here
      // are independently enumerated in face coordinates, without cube-face transitions.
      for (const level of [26, 30])
        for (let face = 0; face < 6; face++)
          keys.push(getS2TopologyCell({face, ij: [2 ** (level - 1), 2 ** (level - 1)], level}));
      keys.push(...INVALID_KEYS);
      while (keys.length < 257) keys.push(keys.length % 3 === 0 ? 0n : keys[keys.length % 17]);
      for (const wordOrder of ['little-endian', 'high-low'] as const)
        for (const withMask of [false, true])
          await checkRows({type, k: radius}, keys, wordOrder, withMask);
    }, 180_000);

for (const resolution of [0, 1, 15, 30])
  it(`S2 parents at level ${resolution} match independent Hilbert descent on mixed-level batches`, async () => {
    const keys = [...S2_TOPOLOGY_REFERENCE.map(row => row.cell), ...INVALID_KEYS];
    for (const wordOrder of ['little-endian', 'high-low'] as const)
      for (const withMask of [false, true])
        await checkRows({type: 'parent', resolution}, keys, wordOrder, withMask);
  }, 180_000);

for (const [inputResolution, resolution] of [
  [0, 0],
  [0, 6],
  [24, 30],
  [29, 30],
  [30, 30]
])
  it(`S2 children from level ${inputResolution} to ${resolution} match recursive math.gl subdivision`, async () => {
    const size = 2 ** inputResolution;
    const keys = Array.from({length: 6}, (_, face) =>
      getS2TopologyCell({face, ij: [size - 1, Math.floor(size / 2)], level: inputResolution})
    );
    keys.push(1n << 60n, 1n, ...INVALID_KEYS);
    for (const wordOrder of ['little-endian', 'high-low'] as const)
      for (const withMask of [false, true])
        await checkRows({type: 'children', inputResolution, resolution}, keys, wordOrder, withMask);
  }, 180_000);
