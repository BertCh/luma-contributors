// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createGPULocalFrame} from '../../../src/analysis-core/contracts/local-frame';
import {CPULocalFrameProjectionOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-local-frame-projection-operation';
import {
  GPULocalFrameProjectionOperation,
  type LocalFrameProjectionOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const kind of ['azimuthal-equidistant', 'lambert-equal-area'] as const) {
  for (const positionsFormat of ['float32x2', 'uint32x4'] as const) {
    for (const edgeCase of [false, true]) {
      it(`projection twins compare coordinates and validity (${kind}, ${positionsFormat}, ${edgeCase ? 'rejected/boundary' : 'ordinary'})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const coordinates = edgeCase
          ? [NaN, 0, 0, Infinity, 2, 0, -1, -1]
          : [0, 0, 0.5, 0.25, 0.123456789, 0.234567891, 1, 1];
        const values =
          positionsFormat === 'uint32x4'
            ? new Uint32Array(new Float64Array(coordinates).buffer)
            : new Float32Array(coordinates);
        const props: LocalFrameProjectionOperationProps = {
          frame: createGPULocalFrame({origin: [0, 0], kind}),
          bounds: [-1, -1, 1, 1],
          positions: 'positions',
          positionsFormat,
          rowCount: 4,
          output: 'projected',
          validity: 'validity'
        };
        const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
        for (const Operation of [
          GPULocalFrameProjectionOperation,
          CPULocalFrameProjectionOperation
        ]) {
          const operation = new Operation(props);
          try {
            results.push(
              await runOperationGraph(device, {
                inputs: {positions: {format: positionsFormat, values}},
                operations: [operation],
                outputs: ['projected', 'validity']
              })
            );
          } finally {
            // runOperationGraph destroys the compiled graph before returning or throwing.
            if (operation instanceof GPULocalFrameProjectionOperation) operation.destroy();
          }
        }
        expect(Array.from(results[0]['validity'])).toEqual(edgeCase ? [0, 0, 0, 1] : [1, 1, 1, 1]);
        expect(Array.from(results[0]['validity'])).toEqual(Array.from(results[1]['validity']));
        expect(results[0]['projected'].length).toBe(8);
        if (!edgeCase) {
          for (const result of results) {
            // The first input is exactly the frame origin, whose projected image is (0, 0).
            expect(Math.abs(result['projected'][0])).toBeLessThanOrEqual(0.01);
            expect(Math.abs(result['projected'][1])).toBeLessThanOrEqual(0.01);
          }
        }
        // 0.01 m patch tolerance plus absolute float32 output rounding within 160 km: 0.03 m.
        results[1]['projected'].forEach((expected, index) => {
          expect(Math.abs(results[0]['projected'][index] - expected)).toBeLessThanOrEqual(0.03);
          if (results[1]['validity'][Math.floor(index / 2)] === 0) {
            expect(results[0]['projected'][index]).toBe(0);
          }
        });
      });
    }
  }
}
