// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUOffsetCurveComponents} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components';
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
import {OFFSET_CURVE_ENDPOINT_CASES} from './offset-curve-components-test-utils';

useCPUExecutorTestHooks();

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
