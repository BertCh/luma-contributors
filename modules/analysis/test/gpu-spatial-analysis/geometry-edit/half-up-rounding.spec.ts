// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUGeometryCleanup,
  getGPUGeometryCleanupParameterValues
} from '../../../src/gpu-spatial-analysis/geometry-edit';
import {getCoverageSnapKeyWGSL} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/coverage-topology-kernels';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const [gridSize, values, expectedCells, expected] of [
  [
    1,
    [0, 8388609, 0.4999999701976776, 0.5, -0.5, -1.5],
    [0, 8388609, 0, 1, 0, -1],
    [0, 8388609, 0, 1, 0, -1]
  ],
  [41, [0, 20.5, -20.5, 41], [0, 1, 0, 1], [0, 41, 0, 41]],
  [7, [0, 58720228, -58720228], [0, 8388604, -8388604], [0, 58720228, -58720228]]
] as const) {
  it(`UGE-03 cleanup and zero-origin coverage keys round exactly at spacing ${gridSize}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      // Literal exact half-up answers, including the negative tie toward positive infinity.
      const positions = rig.input(
        Float32Array.from(values.flatMap(value => [value, 0])),
        'float32x2',
        values.length
      );
      const output = rig.output('float32x2', values.length);
      rig.graph.add(
        new GPUGeometryCleanup({
          geometryType: 'points',
          positions,
          parameters: rig.input(
            getGPUGeometryCleanupParameterValues({gridSize, removeRepeatedPoints: false}),
            'float32',
            4
          ),
          output: {
            positions: output.view,
            status: {count: rig.output('uint32', 1).view, overflow: rig.output('uint32', 1).view}
          }
        })
      );
      const keys = rig.output('uint32', values.length);
      const offsets = rig.input(Uint32Array.of(0, values.length), 'uint32', 2);
      rig.graph.add(
        createWGSLKernelNode(rig.graph, {
          id: 'coverage-key-probe',
          operation: 'coverage-key-probe',
          variant: 'keys',
          invocationCount: values.length,
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            {name: 'ringOffsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'keys', view: keys.view, type: 'u32', access: 'read_write'}
          ],
          declarations: `${IS_FINITE_FLOAT_WGSL}\n${getCoverageSnapKeyWGSL(gridSize, 1)}`,
          body: 'keys[keysOffset + index] = quantise(positions[positionsOffset + index * 2u], getSnapOrigin().x);'
        })
      );
      // Declare both producers and every view before compilation freezes the graph.
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(output.buffer, values.length * 2);
        actual.forEach(value => expect(Number.isFinite(value)).toBe(true));
        expect(actual.map(value => value + 0)).toEqual(expected.flatMap(value => [value, 0]));
        expect(await readUint32(keys.buffer, values.length)).toEqual(
          expectedCells.map(value => (value ^ 0x80000000) >>> 0)
        );
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });
}
