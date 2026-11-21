// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  decodeGPUOffsetCurveComponentsStatus,
  GPUOffsetCurveComponents
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components';
import {
  GPUOffsetCurve,
  getGPUOffsetCurveRowsPerVertex
} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  OFFSET_CURVE_ENDPOINT_CASES,
  PROJECTED_OBLIQUE_OFFSET_CASES,
  expectCompleteObliqueOffset
} from './offset-curve-components-test-utils';

useCPUExecutorTestHooks();

it.each([
  4, 32
])('rejects an unbounded source-edge span with segment capacity %i', segmentCapacity => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const original = {
      positions: createTransientView(graph, 'original', 'float32x2', 4),
      lineOffsets: createTransientView(graph, 'original-offsets', 'uint32', 3)
    };
    const raw = {
      positions: createTransientView(graph, 'raw', 'float32x2', 4),
      lineOffsets: createTransientView(graph, 'raw-offsets', 'uint32', 3)
    };
    memory.write(original.positions, Float32Array.of(0, 0, 2, 0, 5, 0, 7, 0));
    memory.write(raw.positions, Float32Array.of(0, 1, 2, 1, 5, 1, 7, 1));
    memory.write(original.lineOffsets, Uint32Array.of(0, 2, 0xffffffff));
    memory.write(raw.lineOffsets, Uint32Array.of(0, 2, 4));
    const output = {
      positions: createTransientView(graph, 'positions', 'float32x2', 8),
      components: createTransientView(graph, 'components', 'uint32x4', 4),
      sourceEdges: createTransientView(graph, 'edges', 'uint32', 8),
      status: createTransientView(graph, 'status', 'uint32x4', 2)
    };
    const nodes = new GPUOffsetCurveComponents({
      original,
      raw,
      output,
      distance: 1,
      tolerance: 0,
      intersectionCapacity: 4,
      segmentCapacity
    }).getCommandNodes(graph);
    executeCommandNodesOnCPU(nodes, memory);
    // The first line contributes one horizontal component. The second edge range would need
    // billions of reads, so the graph marks iterationLimit and suppresses publication.
    expect(Array.from(memory.readUint32(output.status))).toEqual([0, 0, 1, 2, 0, 0, 1, 0]);
    expect(decodeGPUOffsetCurveComponentsStatus(memory.readUint32(output.status))).toMatchObject({
      iterationLimit: true,
      complete: false
    });
    memory.write(original.lineOffsets, Uint32Array.of(0, 2, 4));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(output.status))).toEqual([2, 4, 2, 4, 0, 0, 0, 0]);
  } finally {
    device.destroy();
  }
});

it.each(
  PROJECTED_OBLIQUE_OFFSET_CASES
)('interpreted projected offset kernels retain both endpoints: $name', fixture => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const original = {
    positions: createTransientView(graph, 'original', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'original-offsets', 'uint32', 2)
  };
  const raw = {
    positions: createTransientView(graph, 'raw', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'raw-offsets', 'uint32', 2)
  };
  memory.write(original.positions, fixture.originalPositions);
  memory.write(raw.positions, fixture.expectedPositions);
  memory.write(original.lineOffsets, Uint32Array.of(0, 2));
  memory.write(raw.lineOffsets, Uint32Array.of(0, 2));
  const output = {
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    components: createTransientView(graph, 'components', 'uint32x4', 4),
    sourceEdges: createTransientView(graph, 'edges', 'uint32', 8),
    status: createTransientView(graph, 'status', 'uint32x4', 2)
  };
  try {
    executeCommandNodesOnCPU(
      new GPUOffsetCurveComponents({
        original,
        raw,
        output,
        distance: 5,
        tolerance: fixture.tolerance,
        intersectionCapacity: 4,
        segmentCapacity: 4
      }).getCommandNodes(graph),
      memory
    );
    expectCompleteObliqueOffset(
      {
        positions: memory.readFloat32(output.positions),
        components: memory.readUint32(output.components),
        sourceEdges: memory.readUint32(output.sourceEdges),
        status: memory.readUint32(output.status)
      },
      fixture.expectedPositions
    );
  } finally {
    device.destroy();
  }
});

it('interpreted multipart offset kernels preserve every feature and part provenance word', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const featureOffsets = createTransientView(graph, 'features', 'uint32', 4);
  const original = {
    positions: createTransientView(graph, 'original', 'float32x2', 6),
    lineOffsets: createTransientView(graph, 'original-lines', 'uint32', 4),
    featureOffsets
  };
  const raw = {
    positions: createTransientView(graph, 'raw', 'float32x2', 6),
    lineOffsets: createTransientView(graph, 'raw-lines', 'uint32', 4),
    featureOffsets
  };
  memory.write(featureOffsets, Uint32Array.of(0, 0, 2, 3));
  memory.write(original.positions, Float32Array.of(0, 0, 2, 0, 5, 0, 7, 0, 10, 0, 12, 0));
  memory.write(raw.positions, Float32Array.of(0, 1, 2, 1, 5, 1, 7, 1, 10, 1, 12, 1));
  memory.write(original.lineOffsets, Uint32Array.of(0, 2, 4, 6));
  memory.write(raw.lineOffsets, Uint32Array.of(0, 2, 4, 6));
  const output = {
    positions: createTransientView(graph, 'positions', 'float32x2', 6),
    components: createTransientView(graph, 'components', 'uint32x4', 3),
    sourceEdges: createTransientView(graph, 'sources', 'uint32', 6),
    status: createTransientView(graph, 'status', 'uint32x4', 2)
  };
  try {
    executeCommandNodesOnCPU(
      new GPUOffsetCurveComponents({
        original,
        raw,
        output,
        distance: 1,
        tolerance: 0,
        intersectionCapacity: 4,
        segmentCapacity: 4
      }).getCommandNodes(graph),
      memory
    );
    // multipart-offset-components.oracle.py: Shapely offsets; part/feature row mapping by hand.
    expect(Array.from(memory.readFloat32(output.positions))).toEqual([
      0, 1, 2, 1, 5, 1, 7, 1, 10, 1, 12, 1
    ]);
    expect(Array.from(memory.readUint32(output.components))).toEqual([
      0, 2, 0, 1, 2, 2, 1, 1, 4, 2, 2, 2
    ]);
    expect(Array.from(memory.readUint32(output.sourceEdges))).toEqual([0, 0, 2, 2, 4, 4]);
    expect(Array.from(memory.readUint32(output.status))).toEqual([3, 6, 3, 6, 0, 0, 0, 0]);
  } finally {
    device.destroy();
  }
});

it.each([
  {
    positions: [0, 0, 4, 4, 0, 4, 4, 0],
    joinStyle: 'round' as const,
    count: 3,
    length: 8.485281374238568
  },
  {
    positions: [0, 0, 6, 0, 6, 4, 3, 4, 3, 1, 0, 1],
    joinStyle: 'mitre' as const,
    count: 1,
    length: 10
  }
])('interpreted generated $joinStyle offset preserves components and length', fixture => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  const original = {
    positions: createTransientView(graph, 'original', 'float32x2', fixture.positions.length / 2),
    lineOffsets: createTransientView(graph, 'original-offsets', 'uint32', 2)
  };
  const rowsPerVertex = getGPUOffsetCurveRowsPerVertex(fixture.joinStyle, 8, 'buffer');
  const raw = {
    positions: createTransientView(
      graph,
      'raw',
      'float32x2',
      original.positions.length * rowsPerVertex
    ),
    lineOffsets: createTransientView(graph, 'raw-offsets', 'uint32', 2)
  };
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  memory.write(parameters, Float32Array.of(0.5, 5, 0, 0));
  memory.write(original.positions, Float32Array.from(fixture.positions));
  memory.write(original.lineOffsets, Uint32Array.of(0, original.positions.length));
  memory.write(raw.lineOffsets, Uint32Array.of(0, raw.positions.length));
  const output = {
    positions: createTransientView(graph, 'positions', 'float32x2', 256),
    components: createTransientView(graph, 'components', 'uint32x4', 32),
    sourceEdges: createTransientView(graph, 'source-edges', 'uint32', 256),
    status: createTransientView(graph, 'status', 'uint32x4', 2)
  };
  const generate = new GPUOffsetCurve({
    ...original,
    parameters,
    output: {positions: raw.positions},
    joinStyle: fixture.joinStyle,
    quadSegments: 8,
    insideTurns: 'buffer'
  });
  const select = new GPUOffsetCurveComponents({
    original,
    raw,
    output,
    rowsPerVertex,
    distance: 0.5,
    tolerance: 0.0025,
    intersectionCapacity: 32,
    segmentCapacity: 128
  });
  const selectNodes = select.getCommandNodes(graph);
  executeCommandNodesOnCPU([...generate.getCommandNodes(graph), ...selectNodes], memory);
  const status = memory.readUint32(output.status);
  const positions = memory.readFloat32(output.positions);
  const components = memory.readUint32(output.components);
  let length = 0;
  for (let component = 0; component < status[0]; component++) {
    const first = components[component * 4];
    const count = components[component * 4 + 1];
    for (let vertex = first + 1; vertex < first + count; vertex++)
      length += Math.hypot(
        positions[vertex * 2] - positions[(vertex - 1) * 2],
        positions[vertex * 2 + 1] - positions[(vertex - 1) * 2 + 1]
      );
  }
  const diagnostic = JSON.stringify({
    status: Array.from(status),
    positions: Array.from(positions.subarray(0, status[1] * 2)),
    components: Array.from(components.subarray(0, status[0] * 4))
  });
  expect(Number.isFinite(length)).toBe(true);
  expect(length, diagnostic).toBeCloseTo(fixture.length, 3);
  expect(status[0], diagnostic).toBe(fixture.count);
  expect(Array.from(status.subarray(4)), diagnostic).toEqual([0, 0, 0, 0]);
});

it.each(OFFSET_CURVE_ENDPOINT_CASES)('interpreted offset kernels preserve $name', fixture => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  const original = {
    positions: createTransientView(
      graph,
      'original',
      'float32x2',
      fixture.originalPositions.length / 2
    ),
    lineOffsets: createTransientView(graph, 'original-offsets', 'uint32', 2)
  };
  const raw = {
    positions: createTransientView(graph, 'raw', 'float32x2', fixture.rawPositions.length / 2),
    lineOffsets: createTransientView(graph, 'raw-offsets', 'uint32', 2)
  };
  memory.write(original.positions, Float32Array.from(fixture.originalPositions));
  memory.write(original.lineOffsets, Uint32Array.of(0, original.positions.length));
  memory.write(raw.positions, Float32Array.from(fixture.rawPositions));
  memory.write(raw.lineOffsets, Uint32Array.of(0, raw.positions.length));
  const output = {
    positions: createTransientView(graph, 'positions', 'float32x2', 16),
    components: createTransientView(graph, 'components', 'uint32x4', 4),
    sourceEdges: createTransientView(graph, 'source-edges', 'uint32', 16),
    status: createTransientView(graph, 'status', 'uint32x4', 2)
  };
  const contributor = new GPUOffsetCurveComponents({
    original,
    raw,
    output,
    distance: fixture.distance,
    tolerance: 0,
    intersectionCapacity: 8,
    segmentCapacity: 16
  });
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(output.status))).toEqual(fixture.status);
  expect(
    Array.from(memory.readFloat32(output.positions).subarray(0, fixture.positions.length))
  ).toEqual(fixture.positions);
  expect(
    Array.from(memory.readUint32(output.components).subarray(0, fixture.components.length))
  ).toEqual(fixture.components);
  expect(
    Array.from(memory.readUint32(output.sourceEdges).subarray(0, fixture.sourceEdges.length))
  ).toEqual(fixture.sourceEdges);
});
