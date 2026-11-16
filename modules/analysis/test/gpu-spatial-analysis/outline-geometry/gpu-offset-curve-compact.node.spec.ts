// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {getCompactOffsetExpected} from './compact-offset-oracle';
import {OFFSET_CURVE_REFERENCE_CASES} from './offset-curve-reference';
import {
  GPUOutlineGeometry,
  getGPUOutlineGeometryParameterValues
} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-outline-geometry';
import {
  GPUOffsetCurve,
  getGPUOffsetCurveParameterValues
} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());

function createProps(graph: GPUCommandGraph, rows = 5, capacity = 8, lineCount = 2) {
  return {
    id: 'compact-offset',
    positions: createTransientView(graph, 'positions', 'float32x2', rows),
    lineOffsets: createTransientView(graph, 'lines', 'uint32', lineCount + 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    joinStyle: 'bevel' as const,
    output: {
      layout: 'compact' as const,
      positions: createTransientView(graph, 'result', 'float32x2', capacity),
      lineOffsets: createTransientView(graph, 'offsets', 'uint32', lineCount + 1),
      sourceRows: createTransientView(graph, 'source-rows', 'uint32', capacity),
      sourceLineIndices: createTransientView(graph, 'source-lines', 'uint32', lineCount),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        vertexCount: createTransientView(graph, 'vertices', 'uint32', 1),
        requiredVertexCount: createTransientView(graph, 'required-vertices', 'uint32', 1)
      }
    }
  };
}

it('emits ordered compact corners and rewrites exact requirements on each frame', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 5, 5);
  const nodes = new GPUOffsetCurve(props).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0, 0, 4, 0, 4, 4, 10, 0, 12, 0]));
  memory.write(props.lineOffsets, new Uint32Array([0, 3, 5]));
  const run = (distance: number) => {
    memory.write(props.parameters, getGPUOffsetCurveParameterValues({distance}));
    executeCommandNodesOnCPU(nodes, memory);
    return {
      positions: Array.from(memory.readFloat32(props.output.positions)),
      offsets: Array.from(memory.readUint32(props.output.lineOffsets)),
      rows: Array.from(memory.readUint32(props.output.sourceRows)),
      lines: Array.from(memory.readUint32(props.output.sourceLineIndices)),
      count: memory.readUint32(props.output.status.count)[0],
      required: memory.readUint32(props.output.status.requiredVertexCount)[0],
      written: memory.readUint32(props.output.status.vertexCount)[0],
      overflow: memory.readUint32(props.output.status.overflow)[0]
    };
  };
  // Outside bevel: four vertices on the first line and two on the second; suffix truncation.
  expect(run(-1)).toEqual({
    positions: [0, -1, 4, -1, 5, 0, 5, 4, 10, -1],
    offsets: [0, 4, 5],
    rows: [0, 1, 1, 2, 3],
    lines: [0, 1],
    count: 2,
    required: 6,
    written: 5,
    overflow: 1
  });
  // Inside corner: the two offset lines intersect at (3, 1).
  expect(run(1)).toEqual({
    positions: [0, 1, 3, 1, 3, 4, 10, 1, 12, 1],
    offsets: [0, 3, 5],
    rows: [0, 1, 2, 3, 4],
    lines: [0, 1],
    count: 2,
    required: 5,
    written: 5,
    overflow: 0
  });
});

it('accepts separate-pass packed status and rejects co-bound aligned overlap', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(
    () =>
      new GPUOffsetCurve({
        ...props,
        output: {
          ...props.output,
          sourceRows: graph.createDataView(props.output.positions.buffer, {
            format: 'uint32',
            length: 8
          })
        }
      })
  ).toThrow(/byte ranges must not overlap/);
  const shared = createTransientView(graph, 'shared-status', 'uint32', 65);
  const count = graph.createDataView(shared.buffer, {format: 'uint32', length: 1});
  const overflow = graph.createDataView(shared.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  expect(
    () =>
      new GPUOffsetCurve({
        ...props,
        output: {...props.output, status: {...props.output.status, count, overflow}}
      })
  ).not.toThrow();
  expect(
    () =>
      new GPUOffsetCurve({
        ...props,
        output: {...props.output, status: {...props.output.status, overflow, vertexCount: count}}
      })
  ).toThrow(/aligned binding byte ranges must not overlap/);
});

it('pre-checks aligned output bindings before allocating compact scratch or nodes', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 1, 16_777_216, 1);
  const backing = createTransientView(graph, 'padded-result', 'float32', 33_554_433);
  props.output.positions = graph.createDataView(backing.buffer, {
    format: 'float32x2',
    length: 16_777_216,
    byteOffset: 4
  });
  const contributor = new GPUOffsetCurve(props);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 134217732 bytes/);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('bounds the required count before a scan can wrap, independently of output capacity', () => {
  const graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPUOffsetCurve({
        ...createProps(graph, 8_372_260, 1, 1),
        joinStyle: 'round',
        quadSegments: 256
      })
  ).toThrow(/positions.length \* rowsPerVertex must be at most 4294967295/);
});

it('keeps exactly two corner pipelines as the number of lines grows', () => {
  for (const lineCount of [1, 10, 100]) {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUOffsetCurve(
      createProps(graph, lineCount * 3, lineCount * 4, lineCount)
    );
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const cornerNodes = nodes.filter(
      node => /-(count|generate)$/.test(node.id) && !node.id.endsWith('-line-count')
    );
    expect(getGPUCommandNodePipelineKeys(cornerNodes).size).toBe(2);
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
      expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
      expect(stripWGSLComments(source)).not.toMatch(/\b(?:nan|inf)\b/i);
    }
  }
});

it('pre-checks the aligned outline output binding before declaring any nodes', () => {
  const constrainedDevice = createNullWebGPUDevice();
  Object.defineProperty(constrainedDevice.limits, 'maxStorageBufferBindingSize', {value: 360});
  const graph = new GPUCommandGraph(constrainedDevice);
  try {
    const positions = createTransientView(graph, 'positions', 'float32x2', 3);
    const parameters = createTransientView(graph, 'parameters', 'float32', 4);
    const backing = createTransientView(graph, 'padded-outline', 'float32', 91);
    const output = graph.createDataView(backing.buffer, {
      format: 'float32x2',
      length: 45,
      byteOffset: 4
    });
    const contributor = new GPUOutlineGeometry({
      id: 'outline-binding',
      geometryType: 'points',
      positions,
      joinSegments: 3,
      parameters,
      output: {positions: output}
    });
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /^outline-binding output\.0 chunk 0 needs 364 bytes/
    );
  } finally {
    constrainedDevice.destroy();
  }
});

it('validates an outline sphere radius at construction before emitting an f32 literal', () => {
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 1);
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  const output = createTransientView(graph, 'outline', 'float32x2', 54);
  expect(
    () =>
      new GPUOutlineGeometry({
        id: 'outline-radius',
        geometryType: 'points',
        positions,
        parameters,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          sphereRadius: 1e100
        },
        output: {positions: output}
      })
  ).toThrow(/^outline-radius spatialContext sphere radius must be/);
});

it('preserves the padded layout and per-corner counts on the original route', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 3, 6, 1);
  const positions = createTransientView(graph, 'fixed-positions', 'float32x2', 6);
  const counts = createTransientView(graph, 'fixed-counts', 'uint32', 3);
  const contributor = new GPUOffsetCurve({...props, output: {positions, counts}});
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes).toHaveLength(1);
  expect(contributor.estimatedCommandNodeCount).toBe(1);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0, 0, 4, 0, 4, 4]));
  memory.write(props.lineOffsets, new Uint32Array([0, 3]));
  memory.write(props.parameters, getGPUOffsetCurveParameterValues({distance: -1}));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readFloat32(positions))).toEqual([
    0, -1, 0, -1, 4, -1, 5, 0, 5, 4, 5, 4
  ]);
  expect(Array.from(memory.readUint32(counts))).toEqual([1, 2, 1]);
});

it('bounds the real upstream scan hierarchy rather than relying on the node scan emulator', () => {
  vi.restoreAllMocks();
  for (const rows of [1, 256, 65_536, 2_000_000]) {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUOffsetCurve(createProps(graph, rows, 1, 1));
    expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
  }
});

it('preserves empty lines and implicitly closed polygon rings in compact offsets', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 4, 4, 3);
  const featureOffsets = createTransientView(graph, 'features', 'uint32', 2);
  const polygonOffsets = createTransientView(graph, 'polygons', 'uint32', 2);
  const contributor = new GPUOffsetCurve({
    ...props,
    geometryType: 'polygons',
    featureOffsets,
    polygonOffsets,
    ringOffsets: props.lineOffsets
  });
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0, 0, 4, 0, 4, 4, 0, 4]));
  memory.write(props.lineOffsets, new Uint32Array([0, 0, 4, 4]));
  memory.write(featureOffsets, new Uint32Array([0, 1]));
  memory.write(polygonOffsets, new Uint32Array([0, 3]));
  memory.write(props.parameters, getGPUOffsetCurveParameterValues({distance: 1}));
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readFloat32(props.output.positions))).toEqual([1, 1, 3, 1, 3, 3, 1, 3]);
  expect(Array.from(memory.readUint32(props.output.lineOffsets))).toEqual([0, 0, 4, 4]);
  expect(Array.from(memory.readUint32(props.output.sourceLineIndices))).toEqual([0, 1, 2]);
  expect(Array.from(memory.readUint32(props.output.sourceRows))).toEqual([0, 1, 2, 3]);
  expect(memory.readUint32(props.output.status.count)[0]).toBe(3);
  expect(memory.readUint32(props.output.status.requiredCount)[0]).toBe(3);
});

it('audits generated identifiers for every join and inside-turn variant', () => {
  for (const joinStyle of ['round', 'mitre', 'bevel'] as const) {
    for (const insideTurns of ['intersection', 'buffer'] as const) {
      const graph = new GPUCommandGraph(device);
      const nodes = new GPUOffsetCurve({
        ...createProps(graph),
        joinStyle,
        insideTurns,
        quadSegments: 256
      }).getCommandNodes(graph);
      for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
        const code = stripWGSLComments(source);
        expect(findReservedIdentifiers(code)).toEqual([]);
        expect(code).not.toMatch(/\b(?:nan|inf)\b/i);
      }
    }
  }
});

it.each([
  1e-50,
  2 ** -149,
  2 ** -127
])('rejects an outline radius that underflows or may flush to zero: %s', sphereRadius => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 1, 54, 1);
  expect(
    () =>
      new GPUOutlineGeometry({
        geometryType: 'points',
        positions: props.positions,
        parameters: props.parameters,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          sphereRadius
        },
        output: {positions: props.output.positions}
      })
  ).toThrow(/spatialContext sphere radius must be/);
});

it('pins all ordered ring corners, retaining a nonrepeated terminal GEOS point', () => {
  const mitre = OFFSET_CURVE_REFERENCE_CASES.find(
    reference => reference.name === 'ring-mitre--0.5'
  )!;
  expect(getCompactOffsetExpected(mitre)).toEqual([
    [-0.5, -0.5],
    [6.5, -0.5],
    [6.5, 4.5],
    [-0.5, 4.5]
  ]);
  const round = getCompactOffsetExpected(
    OFFSET_CURVE_REFERENCE_CASES.find(reference => reference.name === 'ring-round--0.5')!
  );
  expect(round).toHaveLength(36);
  expect(round[0]).toEqual([-0.5, 0]);
  expect(round[8]).toEqual([0, -0.5]);
  expect(round[35]).toEqual([-0.5, 4]);
});

it('rejects non-STORAGE outputs before declaring compact scratch', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const buffer = graph.createTransientBuffer({
    id: 'copy-only',
    byteLength: 4,
    usage: Buffer.COPY_SRC
  });
  const count = graph.createDataView(buffer, {format: 'uint32', length: 1});
  const contributor = new GPUOffsetCurve({
    ...props,
    output: {...props.output, status: {...props.output.status, count}}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/usage must include Buffer.STORAGE/);
  expect(allocate).not.toHaveBeenCalled();
});

it('does not bind or reject unused polygon hierarchy buffers on raw offset and outline routes', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 3, 6, 1);
  const foreign = new GPUCommandGraph(device);
  const buffer = foreign.createTransientBuffer({
    id: 'metadata-only',
    byteLength: 8,
    usage: Buffer.COPY_SRC
  });
  const hierarchy = foreign.createDataView(buffer, {format: 'uint32', length: 2});
  expect(() =>
    new GPUOffsetCurve({
      ...props,
      geometryType: 'polygons',
      ringOffsets: props.lineOffsets,
      featureOffsets: hierarchy,
      polygonOffsets: hierarchy
    }).getCommandNodes(graph)
  ).not.toThrow();
  const outlineGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUOutlineGeometry({
      geometryType: 'polygons',
      joinSegments: 3,
      positions: createTransientView(outlineGraph, 'positions', 'float32x2', 3),
      ringOffsets: createTransientView(outlineGraph, 'rings', 'uint32', 2),
      featureOffsets: hierarchy,
      polygonOffsets: hierarchy,
      parameters: createTransientView(outlineGraph, 'params', 'float32', 4),
      output: {positions: createTransientView(outlineGraph, 'output', 'float32x2', 45)}
    }).getCommandNodes(outlineGraph)
  ).not.toThrow();
});

it('rejects finite JavaScript parameters that would upload infinity as float32', () => {
  expect(() => getGPUOffsetCurveParameterValues({distance: 1e100})).toThrow(
    /GPUOffsetCurve distance/
  );
  expect(() => getGPUOffsetCurveParameterValues({distance: -1e100})).toThrow(
    /GPUOffsetCurve distance/
  );
  expect(() => getGPUOffsetCurveParameterValues({distance: 1, mitreLimit: 1e100})).toThrow(
    /GPUOffsetCurve mitreLimit/
  );
  expect(() => getGPUOutlineGeometryParameterValues({distance: 1e100})).toThrow(
    /GPUOutlineGeometry distance/
  );
});
