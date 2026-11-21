// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {cellArea, getPentagons, gridDisk, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {
  computeCellMeasuresOnCPU,
  CPUCellMeasuresOperation
} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation';
import {
  GPUCellMeasuresOperation,
  type CellMeasuresOperationProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures-operation';
import {GPUS2CellMeasures} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures';
import {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../../../src/analysis-core/contracts/earth';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {quadbinTileToCell} from '../cell-aggregation/cell-aggregation-oracle';
import {s2CellFromFaceIJ} from './cell-indexing-oracle';
import {H3_MEASURES_FIXTURES} from './h3-measures-fixtures';

function getKeys(family: CellMeasuresOperationProps['family']): bigint[] {
  const keys: bigint[] = [];
  if (family === 'h3') {
    for (const resolution of [11, 12, 13, 14, 15]) {
      for (const pentagon of getPentagons(resolution).filter(
        (_, index) => index === 0 || index === 7
      ))
        keys.push(...gridDisk(pentagon, 2).map(key => BigInt(`0x${key}`)));
      keys.push(BigInt(`0x${latLngToCell(37.7749, -122.4194, resolution)}`));
    }
    keys.push(...H3_MEASURES_FIXTURES.map(([key]) => BigInt(`0x${key}`)));
    // Longitude-seam regression: the binary64 twin projects vectors without inverse trig.
    keys.push(0x8ff29380e0d0cc4n);
  } else {
    for (const level of family === 's2' ? [0, 1, 10, 26, 27, 28, 29, 30] : [0, 1, 10, 25, 26]) {
      const size = 2 ** level;
      for (let face = 0; face < (family === 's2' ? 6 : 1); face++)
        for (const [column, row] of [
          [0, 0],
          [size - 1, size - 1],
          [Math.max(0, size / 2 - 1), Math.floor(size / 2)],
          [Math.floor(size / 2), Math.max(0, size / 2 - 1)],
          [Math.floor(size * 0.73), Math.floor(size * 0.18)]
        ])
          keys.push(
            family === 's2'
              ? s2CellFromFaceIJ(face, column * 2 ** (30 - level), row * 2 ** (30 - level), level)
              : quadbinTileToCell(column, row, level)
          );
    }
  }
  keys.push(0n, 1n, 0xffffffffffffffffn);
  // Non-workgroup-multiple length; alternating empty slots exercise all row-aligned outputs.
  while (keys.length < 513) keys.push(keys.length % 3 === 0 ? 0n : keys[keys.length % 17]);
  return keys;
}

function packKeys(keys: bigint[], wordOrder: CellMeasuresOperationProps['wordOrder']): Uint32Array {
  return Uint32Array.from(
    keys.flatMap(key => {
      const words = [Number(key & 0xffffffffn), Number(key >> 32n)];
      return wordOrder === 'high-low' ? words.reverse() : words;
    })
  );
}

it('H3 fp64 subtracts outside fan triangles at concave face-distortion vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // This seven-vertex Class III hexagon has a negative final triangle in a fan anchored
  // on vertex zero. Summing triangle magnitudes gave 1348.88835 m², 3.6377% too large.
  const key = '8b0800000002fff';
  const cells = packKeys([BigInt(`0x${key}`)], 'little-endian');
  const result = await runOperationGraph(device, {
    inputs: {cells: {format: 'uint32x2', values: cells}},
    operations: [
      new GPUCellMeasuresOperation({
        family: 'h3',
        precision: 'double-single',
        rowCount: 1,
        cells: 'cells',
        output: {areas: 'areas', edgeCounts: 'edgeCounts'}
      })
    ],
    outputs: ['areas', 'edgeCounts']
  });
  // Compiled H3 supplies an independent spherical-area oracle, beyond the CPU twin.
  const expected = cellArea(key, 'm2');
  expect(expected).toBeCloseTo(1301.5418132545, 6);
  expect(result['edgeCounts'][0]).toBe(7);
  expect(Math.abs(result['areas'][0] / expected - 1)).toBeLessThan(2e-5);
}, 180_000);

for (const family of ['h3', 's2', 'quadbin'] as const)
  for (const units of ['meters', 'kilometers'] as const)
    for (const wordOrder of ['little-endian', 'high-low'] as const)
      it(`${family} fp64 matches the CPU twin on fine cells, empty slots, ordered edges and stride truncation (${units}, ${wordOrder})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const keys = getKeys(family);
        const cells = packKeys(keys, wordOrder);
        for (const maximumEdgeCount of [family === 'h3' ? 6 : 4, 10]) {
          const props: CellMeasuresOperationProps = {
            family,
            precision: 'double-single',
            cells: 'cells',
            rowCount: keys.length,
            wordOrder,
            units,
            maximumEdgeCount,
            output: {
              areas: 'areas',
              perimeters: 'perimeters',
              edgeLengths: 'edgeLengths',
              edgeCounts: 'edgeCounts',
              pentagons: 'pentagons'
            }
          };
          const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
          for (const Operation of [GPUCellMeasuresOperation, CPUCellMeasuresOperation])
            results.push(
              await runOperationGraph(device, {
                inputs: {cells: {format: 'uint32x2', values: cells}},
                operations: [new Operation(props)],
                outputs: Object.values(props.output) as string[]
              })
            );
          for (const port of ['edgeCounts', 'pentagons'])
            expect(Array.from(results[0][port]), port).toEqual(Array.from(results[1][port]));
          for (const port of ['areas', 'perimeters', 'edgeLengths'])
            results[1][port].forEach((expected, index) => {
              const actual = results[0][port][index];
              if (expected === 0) expect(actual, `${port}[${index}]`).toBe(0);
              else {
                expect(Number.isFinite(actual), `${port}[${index}] finite`).toBe(true);
                expect(
                  Math.abs(actual / expected - 1),
                  `${port}[${index}] expected ${expected}, received ${actual}`
                ).toBeLessThan(family === 'h3' ? 2e-5 : 3e-6);
              }
            });
        }
      }, 180_000);

it('unit-sphere S2 fp64 preserves valid/invalid statuses and agrees with the binary64 oracle at levels 26–30', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const keys = getKeys('s2');
  const cells = packKeys(keys, 'little-endian');
  const expected = computeCellMeasuresOnCPU(cells, {family: 's2'});
  const radius = GPU_EARTH_AUTHALIC_RADIUS_METERS;
  const result = await runOperationGraph(device, {
    inputs: {cells: {format: 'uint32x2', values: cells}},
    operations: [
      new GPUContributorOperation({
        id: 's2-unit-sphere-fp64',
        inputs: {cells: 'cells'},
        outputs: {
          measures: {value: 'measures', format: 'float32x2', length: keys.length},
          status: {value: 'status', format: 'uint32', length: keys.length}
        },
        createContributor: getView =>
          new GPUS2CellMeasures({
            precision: 'double-single',
            cells: getView<'uint32x2'>('cells'),
            output: {
              measures: getView<'float32x2'>('measures'),
              status: getView<'uint32'>('status')
            }
          })
      })
    ],
    outputs: ['measures', 'status']
  });
  keys.forEach((_, row) => {
    if (expected.edgeCounts[row] === 0) {
      expect(result['status'][row]).toBe(1);
      expect(Number.isNaN(result['measures'][2 * row])).toBe(true);
      expect(Number.isNaN(result['measures'][2 * row + 1])).toBe(true);
    } else {
      expect(result['status'][row]).toBe(0);
      expect(
        Math.abs((result['measures'][2 * row] * radius ** 2) / expected.areas[row] - 1)
      ).toBeLessThan(3e-6);
      expect(
        Math.abs((result['measures'][2 * row + 1] * radius) / expected.perimeters[row] - 1)
      ).toBeLessThan(3e-6);
    }
  });
}, 180_000);
