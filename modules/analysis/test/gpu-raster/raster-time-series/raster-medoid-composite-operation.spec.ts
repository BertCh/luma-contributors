// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURasterMedoidCompositeOperation} from '../../../src/raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation';
import {
  GPURasterMedoidCompositeOperation,
  type RasterMedoidCompositeOperationProps
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-medoid-composite-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const missing of [false, true]) {
  it(`compares every medoid output with missing=${missing} (tolerance zero)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: RasterMedoidCompositeOperationProps = {
      cellCount: 1,
      sliceCount: 4,
      bandCount: 2,
      values: 'values',
      mask: 'mask',
      output: {composite: 'composite', sliceIndex: 'sliceIndex', status: 'status'}
    };
    const outputs = ['composite', 'sliceIndex', 'status'];
    const results = [];
    for (const Operation of [GPURasterMedoidCompositeOperation, CPURasterMedoidCompositeOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            values: {format: 'float32', values: new Float32Array([0, 0, 3, 0, 0, 4, 1, 1])},
            mask: {format: 'uint32', values: new Uint32Array(4).fill(missing ? 0 : 9)}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    for (const output of outputs)
      expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
  });
}

for (const values of [
  [0.1, 0.2, 0.3, 0.4],
  [0.1, 0.3, 0.2, 0.4],
  ...[1, 1e-25, 1e20].map(scale => [0, scale, 2 * scale])
]) {
  it(`URT-14 TWN-41 medoid operation preserves literal provenance for ${values}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [
      GPURasterMedoidCompositeOperation,
      CPURasterMedoidCompositeOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {values: {format: 'float32', values: Float32Array.from(values)}},
        operations: [
          new Operation({
            cellCount: 1,
            sliceCount: values.length,
            bandCount: 1,
            values: 'values',
            output: {composite: 'composite', sliceIndex: 'sliceIndex', status: 'status'}
          })
        ],
        outputs: ['composite', 'sliceIndex', 'status']
      });
      // For odd triples the middle value is unique; for even stacks the earliest member of
      // the closed median interval is source row 1, including the permuted middle pair.
      expect(Array.from(result['sliceIndex'])).toEqual([1]);
      expect(Array.from(result['composite'])).toEqual([Math.fround(values[1])]);
      expect(Array.from(result['status'])).toEqual([0]);
    }
  });
}

for (const fixture of [
  {name: 'B4 small leading scalar', values: [7e-12, 1, 3, 5], bandCount: 1, expected: [1]},
  {
    name: 'B5 constant second band',
    values: [0.1, 0.875, 0.2, 0.875, 0.3, 0.875, 0.4, 0.875],
    bandCount: 2,
    expected: [Math.fround(0.2), 0.875]
  },
  {
    name: 'S1 large constant first band',
    values: [1e24, 0, 1e24, 1, 1e24, 2],
    bandCount: 2,
    expected: [Math.fround(1e24), 1]
  }
]) {
  it(`review ${fixture.name} medoid provenance agrees in both operation forms`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [
      GPURasterMedoidCompositeOperation,
      CPURasterMedoidCompositeOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {values: {format: 'float32', values: Float32Array.from(fixture.values)}},
        operations: [
          new Operation({
            cellCount: 1,
            sliceCount: fixture.values.length / fixture.bandCount,
            bandCount: fixture.bandCount,
            values: 'values',
            output: {composite: 'composite', sliceIndex: 'sliceIndex', status: 'status'}
          })
        ],
        outputs: ['composite', 'sliceIndex', 'status']
      });
      expect(Array.from(result['composite'])).toEqual(fixture.expected);
      expect(Array.from(result['sliceIndex'])).toEqual([1]);
      expect(Array.from(result['status'])).toEqual([0]);
    }
  });
}
