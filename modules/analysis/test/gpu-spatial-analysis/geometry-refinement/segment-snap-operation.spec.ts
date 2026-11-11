// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSegmentSnapOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation';
import {
  GPUSegmentSnapOperation,
  type SegmentSnapOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const invalid of [false, true])
  it(`compares every segment-snap output (${invalid ? 'invalid reference' : 'two lines, ties, insertions and overflow'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SegmentSnapOperationProps = {
      positions: 'positions',
      lineOffsets: 'offsets',
      referencePositions: 'references',
      rowCount: 4,
      referenceCount: 7,
      lineCount: 2,
      tolerance: 0.6,
      maximumVerticesPerLine: 5,
      output: {positions: 'snapped', sourceEdges: 'edges', counts: 'counts', validity: 'validity'}
    };
    const references = new Float32Array([0, 0, 2, 0.5, 2, -0.5, 6, 0, 8, 0, 10, 0, 2, 0.5]);
    if (invalid) references[0] = NaN;
    const outputs = ['snapped', 'edges', 'counts', 'validity'];
    const results = [];
    for (const Operation of [GPUSegmentSnapOperation, CPUSegmentSnapOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            positions: {
              format: 'float32x2',
              values: new Float32Array([0.2, 0, 6.2, 0, 0.2, 0, 10.1, 0])
            },
            offsets: {format: 'uint32', values: new Uint32Array([0, 2, 4])},
            references: {format: 'float32x2', values: references}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    // All coordinates are copied from input/reference rows and all other fields are integer:
    // compare exactly, numerically (including zeros). No floating-point tolerance is needed.
    for (const port of outputs) {
      expect(results[0][port].length).toBe(results[1][port].length);
      for (let index = 0; index < results[1][port].length; index++)
        expect(results[0][port][index] === results[1][port][index]).toBe(true);
    }
  });
