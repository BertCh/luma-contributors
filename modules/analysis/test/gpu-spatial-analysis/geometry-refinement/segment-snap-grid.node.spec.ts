// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GPUGridIndex,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {GPUSegmentSnap} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap';
import {computeSegmentSnapOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  countDistinctPipelines
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
function createProps(graph: GPUCommandGraph, lines = 1, candidateCapacity = 16) {
  return {
    id: 'snap-grid',
    candidateCapacity,
    tolerance: 0.6,
    maximumVerticesPerLine: 8,
    positions: createTransientView(graph, 'positions', 'float32x2', lines * 2),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', lines + 1),
    referencePositions: createTransientView(graph, 'references', 'float32x2', 7),
    output: {
      positions: createTransientView(graph, 'output', 'float32x2', lines * 8),
      sourceEdges: createTransientView(graph, 'edges', 'uint32', lines * 8),
      counts: createTransientView(graph, 'counts', 'uint32x4', lines),
      validity: createTransientView(graph, 'validity', 'uint32', lines),
      candidateOverflow: createTransientView(graph, 'candidate-overflow', 'uint32', lines)
    }
  };
}

function mockGridIndex() {
  // The upstream grid scan uses workgroup barriers, which the serial WGSL interpreter cannot
  // emulate. This stand-in implements only its public cell-offset/ID contract in test memory.
  return vi.spyOn(GPUGridIndex.prototype, 'getCommandNodes').mockImplementation(function (
    this: GPUGridIndex
  ) {
    const index = this;
    const views = [
      index.positions,
      index.cellOffsets,
      index.objectIds,
      index.count,
      index.overflow
    ] as GraphDataView[];
    return [
      {
        id: 'cpu-grid-index',
        type: 'compute' as const,
        resources: views.map(view => ({buffer: view, usage: 'storage-read-write' as const})),
        compile: () => ({
          encode: (context: {getBuffer: (view: GraphDataView) => {backing: ArrayBuffer}}) => {
            const words = (view: GraphDataView) =>
              new Uint32Array(
                context.getBuffer(view).backing,
                view.byteOffset,
                (view.length * view.rowByteLength) / 4
              );
            const positions = new Float32Array(
              context.getBuffer(views[0]).backing,
              views[0].byteOffset,
              views[0].length * 2
            );
            const cells = Array.from({length: 256 * 256}, () => [] as number[]);
            for (let row = 0; row < views[0].length; row++) {
              const horizontal = positions[row * 2];
              const vertical = positions[row * 2 + 1];
              if (horizontal < 0 || horizontal > 1 || vertical < 0 || vertical > 1) continue;
              cells[
                Math.min(255, Math.floor(vertical * 256)) * 256 +
                  Math.min(255, Math.floor(horizontal * 256))
              ].push(row);
            }
            let total = 0;
            const offsets = words(views[1]);
            const ids = words(views[2]);
            cells.forEach((cell, row) => {
              offsets[row] = total;
              ids.set(cell, total);
              total += cell.length;
            });
            offsets[cells.length] = total;
            words(views[3])[0] = total;
            words(views[4])[0] = 0;
          }
        })
      }
    ] as unknown as readonly GPUCommandNode<unknown>[];
  });
}

it('uses a fixed set of shaders as the number of line queries grows and audits every identifier', () => {
  const counts = [];
  for (const lines of [256, 257, 769]) {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUSegmentSnap(createProps(graph, lines));
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const sources = getGPUCommandNodeKernelSources(nodes);
    counts.push(new Set(sources.map(({source}) => source)).size);
    expect(
      countDistinctPipelines(nodes.filter(node => /batch-|insert-|candidate-status-/.test(node.id)))
    ).toBe(4);
    for (const {source} of sources)
      expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
  }
  expect(new Set(counts).size).toBe(1);
});

it.each(
  [1, 16].flatMap(candidateCapacity =>
    (['canonical', 'legacy', 'both'] as const).map(spelling => ({candidateCapacity, spelling}))
  )
)('matches independently pinned geometry with capacity $candidateCapacity ($spelling status)', ({
  candidateCapacity,
  spelling
}) => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 1, candidateCapacity);
  if (spelling === 'canonical') {
    const packed = createTransientView(graph, 'packed-references-and-diagnostic', 'uint32', 15);
    props.referencePositions = graph.createDataView(packed.buffer, {
      format: 'float32x2',
      length: 7
    });
    props.output.candidateOverflow = graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 56
    });
  }
  const grid = mockGridIndex();
  const {candidateOverflow, ...outputs} = props.output;
  const status = {
    candidateOverflow: graph.createDataView(candidateOverflow.buffer, {
      format: 'uint32',
      length: candidateOverflow.length,
      byteOffset: candidateOverflow.byteOffset
    })
  };
  const output =
    spelling === 'legacy'
      ? props.output
      : {
          ...outputs,
          status,
          ...(spelling === 'both' ? {candidateOverflow} : {})
        };
  const nodes = new GPUSegmentSnap({...props, output}).getCommandNodes(graph);
  grid.mockRestore();
  const memory = createCPUGraphMemory();
  const positions = new Float32Array([0.2, 0, 6.2, 0]);
  const references = new Float32Array([0, 0, 2, 0.5, 2, -0.5, 6, 0, 8, 0, 10, 0, 2, 0.5]);
  memory.write(props.positions, positions);
  memory.write(props.referencePositions, references);
  memory.write(props.lineOffsets, new Uint32Array([0, 2]));
  executeCommandNodesOnCPU(nodes, memory);
  expect(memory.readFloat32(props.referencePositions)).toEqual(references);
  expect(Array.from(memory.readFloat32(props.output.positions)).slice(0, 8)).toEqual([
    0, 0, 2, 0.5, 2, -0.5, 6, 0
  ]);
  expect(Array.from(memory.readUint32(props.output.counts))).toEqual([4, 4, 0, 0]);
  expect(Array.from(memory.readUint32(props.output.sourceEdges)).slice(0, 4)).toEqual([0, 0, 0, 0]);
  expect(memory.readUint32(props.output.candidateOverflow)[0]).toBe(
    candidateCapacity === 1 ? 1 : 0
  );
  const oracle = computeSegmentSnapOnCPU(
    {positions, referencePositions: references, lineOffsets: new Uint32Array([0, 2])},
    props
  );
  expect(Array.from(oracle.positions).slice(0, 8)).toEqual([0, 0, 2, 0.5, 2, -0.5, 6, 0]);
});

it('checks nested grid and candidate scratch before allocating resources', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 1, 33_554_432);
  const contributor = new GPUSegmentSnap({...props, searchStructure: 'grid'});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/candidates.*bytes/);
  expect(allocate).not.toHaveBeenCalled();
});

it('inserts collinear references without overflowing squared f32 distances', () => {
  const graph = new GPUCommandGraph(device);
  const props = {...createProps(graph), searchStructure: 'brute-force' as const, tolerance: 0};
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0, 0, 4e20, 0]));
  memory.write(
    props.referencePositions,
    new Float32Array([1e20, 0, 2e20, 0, 3e20, 0, 0, 0, 4e20, 0, 1e20, 0, 2e20, 0])
  );
  memory.write(props.lineOffsets, new Uint32Array([0, 2]));
  executeCommandNodesOnCPU(new GPUSegmentSnap(props).getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(props.output.counts))).toEqual([5, 5, 0, 0]);
  expect(Array.from(memory.readFloat32(props.output.positions)).slice(0, 10)).toEqual(
    Array.from(new Float32Array([0, 0, 1e20, 0, 2e20, 0, 3e20, 0, 4e20, 0]))
  );
});

it('batches large default and explicit grid layouts without an exhaustive fallback', () => {
  for (const searchStructure of [undefined, 'grid'] as const) {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 40_000, 1024);
    const nodes = new GPUSegmentSnap({...props, searchStructure}).getCommandNodes(graph);
    expect(nodes.filter(node => node.id.includes('-insert-'))).toHaveLength(
      Math.ceil(40_000 / 256)
    );
    const sources = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id.includes('-insert-'))
    );
    expect(new Set(sources.map(({source}) => source)).size).toBe(1);
    expect(countDistinctPipelines(nodes.filter(node => node.id.includes('-insert-')))).toBe(1);
    for (const node of nodes.filter(node => node.id.includes('-insert-')))
      expect(node.workload?.maximumWorkgroupCount).toBe(1);
  }
});

it.each([
  'grid',
  'brute-force'
] as const)('retains translated collinear references on the %s route', searchStructure => {
  const graph = new GPUCommandGraph(device);
  const props = {...createProps(graph), searchStructure, tolerance: 0};
  const grid = mockGridIndex();
  const nodes = new GPUSegmentSnap(props).getCommandNodes(graph);
  grid.mockRestore();
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([-100, 0, -94, 0]));
  memory.write(props.lineOffsets, new Uint32Array([0, 2]));
  memory.write(
    props.referencePositions,
    new Float32Array([-99, 0, -99, 0, -99, 0, -99, 0, -99, 0, -99, 0, -99, 0])
  );
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(props.output.counts))).toEqual([3, 3, 0, 0]);
  expect(Array.from(memory.readFloat32(props.output.positions)).slice(0, 6)).toEqual([
    -100, 0, -99, 0, -94, 0
  ]);
});

it('bounds every exhaustive overflow route and reports incomplete geometry on both twins', () => {
  const graph = new GPUCommandGraph(device);
  const props = {
    ...createProps(graph, 1, 1),
    referencePositions: createTransientView(graph, 'broad-references', 'float32x2', 300)
  };
  expect(() => new GPUSegmentSnap({...props, searchStructure: 'brute-force'})).toThrow(
    /MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK/
  );
  const grid = mockGridIndex();
  const nodes = new GPUSegmentSnap(props).getCommandNodes(graph);
  grid.mockRestore();
  const memory = createCPUGraphMemory();
  const positions = new Float32Array([0, 0, 6, 0]);
  const references = Float32Array.from({length: 600}, (_, index) => (index % 2 === 0 ? 2 : 0));
  const offsets = new Uint32Array([0, 2]);
  memory.write(props.positions, positions);
  memory.write(props.lineOffsets, offsets);
  memory.write(props.referencePositions, references);
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(props.output.counts))).toEqual([0, 0, 1, 0]);
  expect(Array.from(memory.readUint32(props.output.validity))).toEqual([4]);
  expect(Array.from(memory.readUint32(props.output.candidateOverflow))).toEqual([1]);
  const cpu = computeSegmentSnapOnCPU(
    {positions, lineOffsets: offsets, referencePositions: references},
    props
  );
  expect(Array.from(cpu.counts)).toEqual([0, 0, 1, 0]);
  expect(Array.from(cpu.candidateOverflow)).toEqual([1]);
});

it('declares COPY_DST on reusable batch records and adds the complete graph', () => {
  const graph = new GPUCommandGraph(device);
  expect(() => graph.add(new GPUSegmentSnap(createProps(graph, 257)))).not.toThrow();
});

it('publishes the same source/count/overflow results after reusing records for the next batch', () => {
  const graph = new GPUCommandGraph(device);
  const props = {...createProps(graph, 257, 16), tolerance: 0};
  const grid = mockGridIndex();
  const nodes = new GPUSegmentSnap(props).getCommandNodes(graph);
  grid.mockRestore();
  const positions = new Float32Array(257 * 4);
  for (let line = 0; line < 257; line++) positions.set([-100, 0, -94, 0], line * 4);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, positions);
  memory.write(
    props.lineOffsets,
    Uint32Array.from({length: 258}, (_, line) => line * 2)
  );
  memory.write(
    props.referencePositions,
    new Float32Array([-99, 0, -99, 0, -99, 0, -99, 0, -99, 0, -99, 0, -99, 0])
  );
  executeCommandNodesOnCPU(nodes, memory);
  for (const line of [0, 255, 256]) {
    expect(
      Array.from(memory.readUint32(props.output.counts)).slice(line * 4, line * 4 + 4)
    ).toEqual([3, 3, 0, 0]);
    expect(
      Array.from(memory.readFloat32(props.output.positions)).slice(line * 16, line * 16 + 6)
    ).toEqual([-100, 0, -99, 0, -94, 0]);
    expect(
      Array.from(memory.readUint32(props.output.sourceEdges)).slice(line * 8, line * 8 + 3)
    ).toEqual([line * 2, line * 2, line * 2]);
  }
});

it('keeps broad candidates when a finite tolerance overflows the expanded query box', () => {
  const graph = new GPUCommandGraph(device);
  const props = {...createProps(graph), tolerance: 3e38};
  const grid = mockGridIndex();
  const nodes = new GPUSegmentSnap(props).getCommandNodes(graph);
  grid.mockRestore();
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0.2, 0, 6.2, 0]));
  memory.write(props.lineOffsets, new Uint32Array([0, 2]));
  memory.write(
    props.referencePositions,
    new Float32Array([0, 0, 2, 0.5, 2, -0.5, 6, 0, 8, 0, 10, 0, 2, 0.5])
  );
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(props.output.counts))).toEqual([4, 4, 0, 0]);
  expect(Array.from(memory.readFloat32(props.output.positions)).slice(0, 8)).toEqual([
    0, 0, 2, 0.5, 2, -0.5, 6, 0
  ]);
});
