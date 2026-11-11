// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUOffsetCurveComponentsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation';
import {
  GPUOffsetCurveComponentsOperation,
  type OffsetCurveComponentsOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of ['ordinary', 'bevel', 'circle', 'overflow', 'empty'] as const) {
  it(`offset components compare every CPU/GPU output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: OffsetCurveComponentsOperationProps = {
      id: `components-${scene}`,
      original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
      raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
      lineCount: 2,
      distance: 1,
      tolerance: 0,
      intersectionCapacity: 32,
      segmentCapacity: 32,
      vertexCapacity: scene === 'overflow' ? 4 : 16,
      componentCapacity: scene === 'overflow' ? 2 : 4,
      output: {
        positions: 'positions',
        components: 'components',
        sourceEdges: 'sourceEdges',
        status: 'status'
      }
    };
    const inputs = {
      originalPositions: {
        format: 'float32x2' as const,
        values: Float32Array.of(0, 0, 6, 0, 10, 0, 16, 0)
      },
      originalOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2, 4)},
      rawPositions: {
        format: 'float32x2' as const,
        values:
          scene === 'empty'
            ? Float32Array.of(0, 0.5, 6, 0.5, 10, 0.5, 16, 0.5)
            : Float32Array.of(
                0,
                1,
                3,
                1,
                3,
                2,
                3,
                1,
                6,
                1,
                10,
                1,
                12,
                1,
                12,
                0,
                14,
                0,
                14,
                1,
                16,
                1
              )
      },
      rawOffsets: {
        format: 'uint32' as const,
        values: scene === 'empty' ? Uint32Array.of(0, 2, 4) : Uint32Array.of(0, 5, 11)
      }
    };
    if (scene === 'bevel') {
      props.distance = -1;
      props.rowsPerVertex = 3;
      inputs.originalPositions.values = Float32Array.of(0, 0, 4, 0, 4, 4, 10, 0, 14, 0, 14, 4);
      inputs.originalOffsets.values = Uint32Array.of(0, 3, 6);
      // Exact padded raw GPUOffsetCurve bevels of two left turns, offset to their right.
      inputs.rawPositions.values = Float32Array.of(
        0,
        -1,
        0,
        -1,
        0,
        -1,
        4,
        -1,
        5,
        0,
        5,
        0,
        5,
        4,
        5,
        4,
        5,
        4,
        10,
        -1,
        10,
        -1,
        10,
        -1,
        14,
        -1,
        15,
        0,
        15,
        0,
        15,
        4,
        15,
        4,
        15,
        4
      );
      inputs.rawOffsets.values = Uint32Array.of(0, 9, 18);
    }
    if (scene === 'circle') {
      inputs.rawPositions.values = Float32Array.of(-0.75, -2, -0.75, 2, 9.25, -2, 9.25, 2);
      inputs.rawOffsets.values = Uint32Array.of(0, 2, 4);
    }
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUOffsetCurveComponentsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUOffsetCurveComponentsOperation(props)],
      outputs
    });
    for (const port of ['components', 'sourceEdges', 'status'])
      expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
    // One final f32 rounding vs shader intermediates: 2e-5 absolute planar-unit tolerance.
    expect(gpu['positions'].length).toBe(cpu['positions'].length);
    cpu['positions'].forEach((value, index) =>
      expect(Math.abs(gpu['positions'][index] - value)).toBeLessThanOrEqual(2e-5)
    );
    expect(Array.from(cpu['status'])).toEqual(
      scene === 'ordinary'
        ? [3, 7, 3, 7, 0, 0, 0, 0]
        : scene === 'bevel'
          ? [2, 8, 2, 8, 0, 0, 0, 0]
          : scene === 'circle'
            ? [4, 8, 4, 8, 0, 0, 0, 0]
            : scene === 'overflow'
              ? [1, 3, 3, 7, 1, 0, 0, 0]
              : new Array(8).fill(0)
    );
  });
}

it.each([
  [0, 0],
  [0, 1],
  [2, 0]
])('compares offset vertexCapacity=%i and componentCapacity=%i', async (vertexCapacity, componentCapacity) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: OffsetCurveComponentsOperationProps = {
    id: 'components-zero-capacity',
    original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
    lineCount: 1,
    distance: 1,
    tolerance: 0,
    intersectionCapacity: 4,
    segmentCapacity: 4,
    vertexCapacity,
    componentCapacity,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
  const inputs = {
    originalPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 1, 0)},
    originalOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2)},
    rawPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 1, 1, 1)},
    rawOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2)}
  };
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUOffsetCurveComponentsOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUOffsetCurveComponentsOperation(props)],
    outputs
  });
  // Copied unit coordinates and integer words are exact; compare numeric zeros without sign checks.
  for (const port of outputs) {
    expect(gpu[port].length).toBe(cpu[port].length);
    cpu[port].forEach((value, index) => expect(gpu[port][index] === value).toBe(true));
  }
  for (const result of [gpu, cpu]) {
    expect(Array.from(result['status'])).toEqual([0, 0, 1, 2, 1, 0, 0, 0]);
    expect(Array.from(result['positions'])).toEqual(vertexCapacity === 0 ? [] : [0, 1, 1, 1]);
    expect(Array.from(result['components'])).toEqual(new Array(componentCapacity * 4).fill(0));
    expect(Array.from(result['sourceEdges'])).toEqual(new Array(vertexCapacity).fill(0));
  }
});

it('compares the finite separation sentinel for a singleton original at distance 3.2e38', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: OffsetCurveComponentsOperationProps = {
    id: 'components-separation-sentinel',
    original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
    lineCount: 1,
    distance: 3.2e38,
    tolerance: 0,
    intersectionCapacity: 4,
    segmentCapacity: 4,
    vertexCapacity: 2,
    componentCapacity: 1,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
  const inputs = {
    originalPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 0)},
    originalOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)},
    rawPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 1, 0)},
    rawOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2)}
  };
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUOffsetCurveComponentsOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUOffsetCurveComponentsOperation(props)],
    outputs
  });
  // No original edge is visited, so the finite initializer is below the requested distance and
  // rejects the fragment. Every status word and fresh output row should therefore be zero.
  for (const port of outputs) {
    expect(gpu[port].length).toBe(cpu[port].length);
    for (const result of [gpu, cpu]) expect(result[port].every(value => value === 0)).toBe(true);
  }
});
