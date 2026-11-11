// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPULineLoopRemovalOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-line-loop-removal-operation';
import {
  GPULineLoopRemovalOperation,
  type LineLoopRemovalOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of ['ordinary', 'overflow', 'empty'] as const) {
  it(`loop removal compares every CPU/GPU output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: LineLoopRemovalOperationProps = {
      id: `loops-${scene}`,
      lines: {positions: 'inputPositions', lineOffsets: 'inputOffsets'},
      lineCount: 2,
      vertexCapacity: scene === 'overflow' ? 3 : 12,
      intersectionCapacity: 32,
      output: {
        positions: 'positions',
        lineOffsets: 'lineOffsets',
        sourceEdges: 'sourceEdges',
        status: 'status'
      }
    };
    const inputs = {
      inputPositions: {
        format: 'float32x2' as const,
        values:
          scene === 'ordinary'
            ? Float32Array.of(0, 0, 4, 4, 0, 4, 4, 0, 10, 0, 14, 0, 12, 0, 16, 0)
            : scene === 'overflow'
              ? Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 10, 0, 11, 0)
              : Float32Array.of(0, 0, 0, 0)
      },
      inputOffsets: {
        format: 'uint32' as const,
        values:
          scene === 'ordinary'
            ? Uint32Array.of(0, 4, 8)
            : scene === 'overflow'
              ? Uint32Array.of(0, 5, 7)
              : Uint32Array.of(0, 0, 2)
      }
    };
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPULineLoopRemovalOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPULineLoopRemovalOperation(props)],
      outputs
    });
    for (const port of ['lineOffsets', 'sourceEdges', 'status'])
      expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
    // The crossing is exactly representable and every other coordinate is an input copy.
    // Numeric equality deliberately ignores the sign of zero. All padded rows are checked too.
    expect(gpu['positions'].length).toBe(cpu['positions'].length);
    cpu['positions'].forEach((value, index) =>
      expect(gpu['positions'][index] === value).toBe(true)
    );
    expect(Array.from(cpu['status'])).toEqual(
      scene === 'ordinary' ? [7, 7, 0, 0] : scene === 'overflow' ? [2, 7, 1, 0] : [0, 0, 0, 0]
    );
  });
}

it('compares zero-capacity loop removal with an unsplit two-vertex line', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: LineLoopRemovalOperationProps = {
    id: 'loops-zero-capacity',
    lines: {positions: 'inputPositions', lineOffsets: 'inputOffsets'},
    lineCount: 1,
    vertexCapacity: 0,
    intersectionCapacity: 4,
    output: {
      positions: 'positions',
      lineOffsets: 'lineOffsets',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
  const inputs = {
    inputPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 1, 0)},
    inputOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2)}
  };
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPULineLoopRemovalOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPULineLoopRemovalOperation(props)],
    outputs
  });
  // Only integer status/offset words and empty coordinate buffers: exact numeric comparison.
  for (const port of outputs) expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
  for (const result of [gpu, cpu]) {
    expect(Array.from(result['status'])).toEqual([0, 2, 1, 0]);
    expect(Array.from(result['lineOffsets'])).toEqual([0, 0]);
    expect(result['positions']).toHaveLength(0);
    expect(result['sourceEdges']).toHaveLength(0);
  }
});
