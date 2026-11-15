// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUCellMeasuresOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation';
import {
  GPUCellMeasuresOperation,
  type CellMeasuresOperationProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures-operation';
import {GPUS2CellMeasures} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures';
import {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {quadbinTileToCell} from '../cell-aggregation/cell-aggregation-oracle';
import {s2CellFromFaceIJ} from './cell-indexing-oracle';
import {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../../../src/analysis-core/contracts/earth';
import {GPUCellGeometry} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry';
import {getAngularDistance} from './cell-indexing-oracle';

it('S2 measured edge slots join consecutive GPUCellGeometry boundary vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The reported asymmetric cell, repeated on all six faces. Root-face tests hide reversals.
  const keys = Array.from({length: 6}, (_, face) => 0x0300000000000000n | (BigInt(face) << 61n));
  for (const wordOrder of ['little-endian', 'high-low'] as const)
    for (const maximumEdgeCount of [4, 10]) {
      const cells = Uint32Array.from(
        keys.flatMap(key => {
          const words = [Number(key & 0xffffffffn), Number(key >> 32n)];
          return wordOrder === 'high-low' ? words.reverse() : words;
        })
      );
      const result = await runOperationGraph(device, {
        inputs: {cells: {format: 'uint32x2', values: cells}},
        operations: [
          new GPUCellMeasuresOperation({
            family: 's2',
            cells: 'cells',
            rowCount: keys.length,
            wordOrder,
            // Exercise the default stride and default meter units as reported.
            maximumEdgeCount: maximumEdgeCount === 4 ? undefined : maximumEdgeCount,
            output: {edgeLengths: 'edgeLengths', edgeCounts: 'edgeCounts', perimeters: 'perimeters'}
          }),
          new GPUContributorOperation({
            id: 's2-boundary-geometry',
            inputs: {cells: 'cells'},
            outputs: {
              boundaries: {value: 'boundaries', format: 'float32x2', length: keys.length * 4}
            },
            createContributor: getView =>
              new GPUCellGeometry({
                family: 's2',
                cells: getView<'uint32x2'>('cells'),
                wordOrder,
                output: {boundaries: getView<'float32x2'>('boundaries')}
              })
          })
        ],
        outputs: ['edgeLengths', 'edgeCounts', 'perimeters', 'boundaries']
      });
      keys.forEach((key, row) => {
        const boundary = Array.from({length: 4}, (_, vertex) =>
          Array.from(result['boundaries'].subarray(row * 8 + vertex * 2, row * 8 + vertex * 2 + 2))
        );
        const expectedEdges = boundary.map(
          (first, edge) =>
            getAngularDistance(first, boundary[(edge + 1) % 4]) * GPU_EARTH_AUTHALIC_RADIUS_METERS
        );
        expect(new Set(expectedEdges.map(Math.round)).size, key.toString(16)).toBe(4);
        expect(result['edgeCounts'][row]).toBe(4);
        expectedEdges.forEach((expected, edge) =>
          expect(
            Math.abs(result['edgeLengths'][row * maximumEdgeCount + edge] / expected - 1),
            `${key.toString(16)} edge ${edge}`
          ).toBeLessThan(3e-6)
        );
        expect(
          Math.abs(
            result['perimeters'][row] / expectedEdges.reduce((sum, edge) => sum + edge, 0) - 1
          )
        ).toBeLessThan(3e-6);
        expect(
          Array.from(
            result['edgeLengths'].subarray(row * maximumEdgeCount + 4, (row + 1) * maximumEdgeCount)
          )
        ).toEqual(new Array(maximumEdgeCount - 4).fill(0));
      });
    }
});

for (const family of ['s2', 'quadbin'] as const)
  for (const units of ['meters', 'kilometers'] as const)
    it(`${family} GPU measures match the binary64 twin including fine cells, invalid slots and padding (${units})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const keys: bigint[] = [];
      const levels = family === 's2' ? [0, 1, 10, 26, 27, 28, 29, 30] : [0, 1, 10, 20, 25, 26];
      for (const level of levels) {
        const size = 2 ** level;
        for (let face = 0; face < (family === 's2' ? 6 : 1); face++)
          for (const [first, second] of [
            [0, 0],
            [size - 1, size - 1],
            [Math.floor(size * 0.73), Math.floor(size * 0.18)],
            [Math.floor(size / 2), Math.floor(size / 2)]
          ])
            keys.push(
              family === 's2'
                ? s2CellFromFaceIJ(
                    face,
                    first * 2 ** (30 - level),
                    second * 2 ** (30 - level),
                    level
                  )
                : quadbinTileToCell(first, second, level)
            );
      }
      keys.push(0n, 2n, 0xffffffffffffffffn);
      // A partial final workgroup and repeated empty slots exercise bounded dispatch and padding.
      while (keys.length < 257) keys.push(keys.length % 3 === 0 ? 0n : keys[keys.length % 17]);
      for (const wordOrder of ['little-endian', 'high-low'] as const)
        for (const maximumEdgeCount of [4, 10]) {
          const cells = Uint32Array.from(
            keys.flatMap(key => {
              const words = [Number(key & 0xffffffffn), Number(key >> 32n)];
              return wordOrder === 'high-low' ? words.reverse() : words;
            })
          );
          const props: CellMeasuresOperationProps = {
            family,
            units,
            wordOrder,
            maximumEdgeCount,
            cells: 'cells',
            rowCount: keys.length,
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
            expect(Array.from(results[0][port])).toEqual(Array.from(results[1][port]));
          for (const port of ['areas', 'perimeters', 'edgeLengths'])
            results[1][port].forEach((expected, index) => {
              if (expected === 0) expect(results[0][port][index], `${port}[${index}]`).toBe(0);
              else
                expect(
                  Math.abs(results[0][port][index] / expected - 1),
                  `${port}[${index}], key ${keys[Math.floor(index / (port === 'edgeLengths' ? maximumEdgeCount : 1))].toString(16)}`
                ).toBeLessThan(3e-6);
            });
        }
    });

it('the compatibility S2 class and the unified measures API share the same spherical measures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const keys = [s2CellFromFaceIJ(2, 0, 0, 0), s2CellFromFaceIJ(5, 123456789, 987654321, 30), 0n];
  const cells = Uint32Array.from(
    keys.flatMap(key => [Number(key & 0xffffffffn), Number(key >> 32n)])
  );
  const result = await runOperationGraph(device, {
    inputs: {cells: {format: 'uint32x2', values: cells}},
    operations: [
      new GPUCellMeasuresOperation({
        family: 's2',
        cells: 'cells',
        rowCount: 3,
        output: {areas: 'areas', perimeters: 'perimeters'}
      }),
      new GPUContributorOperation({
        id: 's2-compatibility',
        inputs: {cells: 'cells'},
        outputs: {
          measures: {value: 'measures', format: 'float32x2', length: 3},
          status: {value: 'status', format: 'uint32', length: 3}
        },
        createContributor: getView =>
          new GPUS2CellMeasures({
            cells: getView<'uint32x2'>('cells'),
            output: {
              measures: getView<'float32x2'>('measures'),
              status: getView<'uint32'>('status')
            }
          })
      })
    ],
    outputs: ['areas', 'perimeters', 'measures', 'status']
  });
  const radius = GPU_EARTH_AUTHALIC_RADIUS_METERS;
  for (let row = 0; row < 2; row++) {
    expect(
      Math.abs(result['areas'][row] / (result['measures'][2 * row] * radius ** 2) - 1)
    ).toBeLessThan(2e-7);
    expect(
      Math.abs(result['perimeters'][row] / (result['measures'][2 * row + 1] * radius) - 1)
    ).toBeLessThan(2e-7);
  }
  expect(Array.from(result['status'])).toEqual([0, 0, 1]);
  expect(Number.isNaN(result['measures'][4])).toBe(true);
  expect(result['areas'][2]).toBe(0);
});
