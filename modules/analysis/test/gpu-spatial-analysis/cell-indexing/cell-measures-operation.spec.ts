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
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('compares the resolution-15 longitude-seam cell against the GPU stable face-plane path', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const key = 0x8ff29380e0d0cc4n;
  const cells = Uint32Array.of(Number(key & 0xffffffffn), Number(key >> 32n));
  const props: CellMeasuresOperationProps = {
    family: 'h3',
    cells: 'cells',
    rowCount: 1,
    maximumEdgeCount: 10,
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
      if (expected === 0) expect(results[0][port][index]).toBe(0);
      // This is a single-face hexagon. Its integer-offset GPU path is stable at resolution 15,
      // unlike an inverse-longitude H3 oracle at the seam. Compare positive measures relatively.
      else
        expect(Math.abs(results[0][port][index] / expected - 1), `${port}[${index}]`).toBeLessThan(
          2e-6
        );
    });
  expect(Math.abs(results[1]['areas'][0] / 0.9725676637 - 1)).toBeLessThan(5e-7);
});

for (const units of ['meters', 'kilometers'] as const)
  for (const maximumEdgeCount of [6, 10]) {
    it(`cell measures twins agree on hexagons, pentagons, face crossings and invalid cells (${units}, stride ${maximumEdgeCount})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // Low-resolution fallback boundaries are resolved accurately on the GPU; binary64 is the
      // intended improvement at high resolutions. Relative tolerance 2e-4 covers fallback f32 error.
      const keys = [
        0x808bfffffffffffn,
        0x81083ffffffffffn,
        0x81af7ffffffffffn,
        0x834c02fffffffffn,
        0n,
        1n,
        0xffffffffffffffffn
      ];
      for (const wordOrder of ['little-endian', 'high-low'] as const) {
        const cells = Uint32Array.from(
          keys.flatMap(key => {
            const words = [Number(key & 0xffffffffn), Number(key >> 32n)];
            return wordOrder === 'high-low' ? words.reverse() : words;
          })
        );
        const props: CellMeasuresOperationProps = {
          family: 'h3',
          rowCount: keys.length,
          cells: 'cells',
          maximumEdgeCount,
          units,
          wordOrder,
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
            if (expected === 0) expect(results[0][port][index]).toBe(0);
            else
              expect(
                Math.abs(results[0][port][index] / expected - 1),
                `${port}[${index}]`
              ).toBeLessThan(2e-4);
          });
      }
    });
  }
