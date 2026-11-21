// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createLineOracleHarness} from './line-output-oracle-harness';
import {
  createClipCapacityFixture,
  createClipPaddingFixture,
  createSharedTruncationFixture,
  createClipUncertaintyFixture,
  createSharedUncertaintyFixture
} from './line-clip-output-oracle-fixture';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import type {OracleView} from '../line-merge/line-merge-output-oracle-fixture';

useCPUExecutorTestHooks();

it('shared paths refuses overlapping overflow and uncertainty words', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view: OracleView = (format, length) =>
    createTransientView(graph, `shared-alias-${serial++}`, format, length);
  expect(() => createSharedTruncationFixture(graph, view, true)).toThrow(
    /byte ranges must not overlap/
  );
});

it('shared paths preserves aggregate overflow beside a distinct packed uncertainty word', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view: OracleView = (format, length, values) => {
    const result = createTransientView(graph, `packed-${serial++}`, format, length);
    const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
    memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    return result;
  };
  const fixture = createSharedTruncationFixture(graph, view);
  executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(fixture.count))).toEqual([1]);
  expect(Array.from(memory.readUint32(fixture.requiredCount))).toEqual([2]);
  expect(Array.from(memory.readUint32(fixture.packed))).toEqual([1, 0]);
});

for (const capacity of [1, 2]) {
  it(`clip excludes unused midpoint rows from predicates at intersection capacity ${capacity}`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const memory = createCPUGraphMemory();
    const views: GraphDataView[] = [];
    const createDataView = graph.createDataView.bind(graph);
    const observe = vi.spyOn(graph, 'createDataView').mockImplementation((...parameters) => {
      const result = createDataView(...parameters);
      views.push(result);
      return result;
    });
    let serial = 0;
    const view: OracleView = (format, length, values) => {
      const result = createTransientView(graph, `padding-${serial++}`, format, length);
      const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
      memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
      return result;
    };
    const fixture = createClipPaddingFixture(view, capacity);
    const nodes = fixture.contributor.getCommandNodes(graph);
    observe.mockRestore();
    const transient = (suffix: string) =>
      views.find(candidate => candidate.buffer.id === `line-clip-by-polygon-${suffix}`)!;
    memory.write(transient('sub-counts'), Uint32Array.of(1, 0));
    memory.write(transient('sub-starts'), Uint32Array.of(0, 1));
    memory.write(transient('segment-table'), new Uint32Array(8));
    memory.write(transient('segment-unique-bases'), new Uint32Array(2));
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id === 'line-clip-by-polygon-sub-pieces'),
      memory
    );
    const midpoints = Array.from(memory.readFloat32(transient('sub-midpoints')));
    expect(midpoints.slice(0, 2)).toEqual([-3.5, -3.5]);
    expect(midpoints.slice(2).every(value => !Number.isFinite(value))).toBe(true);
  });
}

for (const capacity of [4, 6]) {
  it(`clip provenance separates caller IDs from part rows at capacity ${capacity}`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const memory = createCPUGraphMemory();
    let serial = 0;
    const view: OracleView = (format, length, values) => {
      const result = createTransientView(graph, `identity-${serial++}`, format, length);
      const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
      memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
      return result;
    };
    const fixture = createClipCapacityFixture(view, capacity, 'parts');
    const truncated = capacity === 4;
    // Independently labeled inputs: part 0 has caller ID 71, part 1 has ID 159.
    // Seed the geometry writer's part-row prefix; execute the real publication
    // nodes that preserve rows separately and convert sourceIds to caller IDs.
    memory.write(fixture.count, Uint32Array.of(truncated ? 1 : 2));
    memory.write(fixture.sourceIds, Uint32Array.of(0, truncated ? 93 : 1));
    const nodes = fixture.contributor.getCommandNodes(graph);
    executeCommandNodesOnCPU(
      nodes.filter(
        node =>
          node.id === 'line-clip-by-polygon-part-rows' ||
          node.id === 'line-clip-by-polygon-source-identities'
      ),
      memory
    );
    expect(Array.from(memory.readUint32(fixture.sourcePartRows!))).toEqual(
      truncated ? [0, 0xffffffff] : [0, 1]
    );
    expect(Array.from(memory.readUint32(fixture.sourceIds))).toEqual(
      truncated ? [71, 0xffffffff] : [71, 159]
    );
  });
}

for (const includeUncertain of [true, false]) {
  it(`shared paths propagates uncertainty with optional scalar present=${includeUncertain}`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const memory = createCPUGraphMemory();
    let serial = 0;
    const view: OracleView = (format, length, values) => {
      const result = createTransientView(graph, `uncertain-${serial++}`, format, length);
      const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
      memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
      return result;
    };
    const fixture = createSharedUncertaintyFixture(view, includeUncertain);
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(fixture.count))).toEqual([0]);
    if (fixture.uncertainCount)
      expect(Array.from(memory.readUint32(fixture.uncertainCount))).toEqual([1]);
    expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([1]);
  });
}

it('clip aggregate flag includes either predicate uncertainty count', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  const views: GraphDataView[] = [];
  const createDataView = graph.createDataView.bind(graph);
  const observe = vi.spyOn(graph, 'createDataView').mockImplementation((...parameters) => {
    const result = createDataView(...parameters);
    views.push(result);
    return result;
  });
  let serial = 0;
  const view: OracleView = (format, length) =>
    createTransientView(graph, `flags-${serial++}`, format, length);
  const fixture = createClipUncertaintyFixture(view, false);
  const nodes = fixture.contributor.getCommandNodes(graph);
  observe.mockRestore();
  const transient = (suffix: string) => {
    const result = views.find(
      candidate => candidate.buffer.id === `line-clip-by-polygon-${suffix}`
    );
    expect(result, suffix).toBeDefined();
    return result!;
  };
  for (const [segmentUncertain, containmentUncertain] of [
    [2, 0],
    [0, 1]
  ]) {
    memory.write(transient('pair-overflow'), Uint32Array.of(0));
    memory.write(transient('containment-overflow'), Uint32Array.of(0));
    memory.write(transient('segment-uncertain'), Uint32Array.of(segmentUncertain));
    memory.write(transient('containment-uncertain'), Uint32Array.of(containmentUncertain));
    // Execute the real aggregation writer with independent nonzero diagnostics;
    // this avoids emulating the containment pipeline's uniform-buffer bindings.
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id === 'line-clip-by-polygon-flags'),
      memory
    );
    expect(Array.from(memory.readUint32(transient('flag-state')))).toEqual([
      1,
      segmentUncertain + containmentUncertain,
      0
    ]);
  }
});

it('clip finalization separates six required vertices from the three that fit', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  const views: GraphDataView[] = [];
  const createDataView = graph.createDataView.bind(graph);
  const observe = vi.spyOn(graph, 'createDataView').mockImplementation((...parameters) => {
    const result = createDataView(...parameters);
    views.push(result);
    return result;
  });
  let serial = 0;
  const view: OracleView = (format, length) =>
    createTransientView(graph, `capacity-${serial++}`, format, length);
  const fixture = createClipCapacityFixture(view, 4, 'counts');
  const nodes = fixture.contributor.getCommandNodes(graph);
  observe.mockRestore();
  const transient = (suffix: string) =>
    views.find(candidate => candidate.buffer.id === `line-clip-by-polygon-${suffix}`)!;
  // Independent prefix sums of two three-vertex clipped pieces: total runs=2,
  // total vertices=6, second run starts at vertex 3. Capacity four fits only run 0.
  const runRanks = new Uint32Array(transient('run-ranks').length);
  runRanks[runRanks.length - 1] = 2;
  const vertexStarts = new Uint32Array(transient('vertex-starts').length);
  vertexStarts[vertexStarts.length - 1] = 6;
  memory.write(transient('run-ranks'), runRanks);
  memory.write(transient('vertex-starts'), vertexStarts);
  memory.write(transient('run-flags'), new Uint32Array(transient('run-flags').length));
  memory.write(transient('vertex-counts'), new Uint32Array(transient('vertex-counts').length));
  const runTable = new Uint32Array(transient('run-table').length);
  runTable.set([0, 0, 3, 1]);
  memory.write(transient('run-table'), runTable);
  memory.write(transient('flag-state'), Uint32Array.of(0, 0, 0));
  // Finalization consumes the production prefix search result. Execute its clear/probe
  // passes against the independent run table rather than leaving that scratch word at zero.
  expect(nodes.some(node => node.id.startsWith('line-clip-by-polygon-capacity-prefix'))).toBe(true);
  executeCommandNodesOnCPU(
    nodes.filter(
      node =>
        node.id.startsWith('line-clip-by-polygon-capacity-prefix') ||
        ['line-clip-by-polygon-finalize', 'line-clip-by-polygon-scalars'].includes(node.id)
    ),
    memory
  );
  expect(Array.from(memory.readUint32(fixture.count))).toEqual([1]);
  expect(Array.from(memory.readUint32(fixture.vertexCount!))).toEqual([3]);
  expect(Array.from(memory.readUint32(fixture.requiredVertexCount!))).toEqual([6]);
  expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([1]);
});

it('oracle harness allocates readable storage buffers and preserves every poison component', async () => {
  const device = createNullWebGPUDevice();
  const harness = createLineOracleHarness(device);
  try {
    for (const format of ['uint32', 'float32', 'float32x2'] as const) {
      const output = harness.view(format, 3);
      // WebGPU readAsync copies from this resource to a mapping buffer. Without
      // COPY_SRC that copy is invalid even though the storage kernel can write it.
      expect(output.buffer.usage & Buffer.COPY_SRC, format).toBe(Buffer.COPY_SRC);
      expect(output.buffer.usage & Buffer.STORAGE, format).toBe(Buffer.STORAGE);
      expect(await harness.read(output)).toEqual(
        new Array(format === 'float32x2' ? 6 : 3).fill(93)
      );
    }
  } finally {
    harness.destroy();
    device.destroy();
  }
});

const UNSUPPORTED_STATUS = [
  'uncertainCount',
  'invalidCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'approximation',
  'ringCount',
  'requiredRingCount'
] as const;

for (const port of ['geometry.sourceIds', ...UNSUPPORTED_STATUS]) {
  it(`clip refuses unwritten pieces.${port} instead of accepting a poisoned output`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
      createTransientView(graph, `clip-${serial++}`, format, length);
    const unsupported = view('uint32', 1);
    expect(
      () =>
        new GPULineClipByPolygon({
          lines: {kind: 'lines', positions: view('float32x2', 2), lineOffsets: view('uint32', 2)},
          polygons: {
            kind: 'polygons',
            positions: view('float32x2', 5),
            featureOffsets: view('uint32', 2),
            polygonOffsets: view('uint32', 2),
            ringOffsets: view('uint32', 2)
          },
          intersectionCapacity: 8,
          candidateCapacity: 32,
          output: {
            pieces: {
              geometry: {
                kind: 'lines',
                positions: view('float32x2', 4),
                lineOffsets: view('uint32', 2),
                ...(port === 'geometry.sourceIds' ? {sourceIds: unsupported} : {})
              },
              sourceIds: view('uint32', 1),
              status: {
                count: view('uint32', 1),
                overflow: view('uint32', 1),
                vertexCount: view('uint32', 1),
                ...(port !== 'geometry.sourceIds' ? {[port]: unsupported} : {})
              }
            }
          }
        })
    ).toThrow(
      `line-clip-by-polygon output.pieces.${port.startsWith('geometry.') ? port : `status.${port}`}`
    );
  });
}
