// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUPairGather} from '../../../src/gpu-spatial-analysis/pair-gather/gpu-pair-gather';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {runExample} from '../../../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.example';
import {getPairGatherCatalogProps} from '../../utils/join-example-constructors';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

useCPUExecutorTestHooks();

// Keep this graph probe on the actual two contributors; unrelated catalog-family modules may
// be under edit in this checkout and are not needed to evaluate the recorded composition.
vi.mock('@luma-contributors/analysis/gpu-spatial-analysis', async () => ({
  GPUPairGather: (await import('../../../src/gpu-spatial-analysis/pair-gather/gpu-pair-gather'))
    .GPUPairGather,
  GPUSpatialPredicateJoin: (
    await import('../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join')
  ).GPUSpatialPredicateJoin
}));

function fixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(format: Format, length = 1) =>
    createTransientView(graph, `final2-${serial++}`, format, length);
  const scalar = () => view('uint32');
  const polygon = (count: number) => ({
    kind: 'polygons' as const,
    positions: view('float32x2', count * 4),
    featureOffsets: view('uint32', count + 1),
    polygonOffsets: view('uint32', count + 1),
    ringOffsets: view('uint32', count + 1)
  });
  return {device, graph, view, scalar, polygon};
}

it('G2 authored catalog gather accepts the recorded predicate composition candidate port', async () => {
  const {device, graph, view, scalar, polygon} = fixture();
  const pairs = {
    leftRows: view('uint32', 2),
    rightRows: view('uint32', 2),
    status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
  };
  try {
    await runExample(graph, {
      parks: polygon(1),
      tracts: polygon(2),
      pairs,
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      parkAreas: view('float32'),
      candidateCapacity: 1
    });
    const props = getPairGatherCatalogProps({
      pairs,
      incomeData: view('float32', 2),
      capacity: 2,
      output: (_id: string, format: 'uint32' | 'float32', length: number) => view(format, length)
    });
    const gather = new GPUPairGather({...props, id: 'catalog-gather'});
    expect(gather.props.output.status.candidateOverflow).toBeDefined();
    expect(gather.getCommandNodes(graph).some(node => node.id.includes('candidate-overflow'))).toBe(
      true
    );
  } finally {
    device.destroy();
  }
});

it('G1 the recorded pair example retains a getter-backed requiredCount writer', async () => {
  const {device, graph, view, scalar, polygon} = fixture();
  const requiredCount = scalar();
  const replacementCount = scalar();
  let reads = 0;
  const status = Object.assign(
    Object.create({
      get requiredCount() {
        return ++reads === 1 ? requiredCount : replacementCount;
      }
    }),
    {
      count: scalar(),
      overflow: scalar(),
      candidateOverflow: scalar()
    }
  );
  const nodes: GPUCommandNode[] = [];
  const original = GPUSpatialPredicateJoin.prototype.getCommandNodes;
  const capture = vi
    .spyOn(GPUSpatialPredicateJoin.prototype, 'getCommandNodes')
    .mockImplementation(function (this: GPUSpatialPredicateJoin, graph) {
      const declared = original.call(this, graph);
      nodes.push(...declared);
      return declared;
    });
  try {
    await runExample(graph, {
      parks: polygon(1),
      tracts: polygon(2),
      pairs: {leftRows: view('uint32', 2), rightRows: view('uint32', 2), status},
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      parkAreas: view('float32'),
      candidateCapacity: 1
    });
    const finalize = nodes.find(node => node.id === 'spatial-predicate-join-finalize');
    expect(
      finalize?.resources?.some(
        resource =>
          'buffer' in resource &&
          resource.buffer === requiredCount &&
          resource.usage !== 'storage-read'
      )
    ).toBe(true);
    expect(reads).toBe(1);
  } finally {
    capture.mockRestore();
    device.destroy();
  }
});

it('G1 getter-backed left mode emits an unmatched row and survives caller mutation', () => {
  const {device, graph, view, scalar} = fixture();
  let how: 'left' | 'inner' = 'left';
  let reads = 0;
  const neighbors = {rows: view('uint32'), counts: view('uint32'), capacity: 1};
  const output = {
    leftRows: view('uint32'),
    rightRows: view('uint32'),
    rowCount: scalar(),
    status: {overflow: scalar()}
  };
  const props = Object.assign(
    Object.create({
      get how() {
        reads++;
        return how;
      }
    }),
    {neighbors, output}
  );
  try {
    const gather = new GPUPairGather(props);
    expect(gather.how).toBe('left');
    expect(reads).toBe(1);
    how = 'inner';
    const memory = createCPUGraphMemory();
    memory.write(neighbors.rows, Uint32Array.of(0xffffffff));
    memory.write(neighbors.counts, Uint32Array.of(0));
    neighbors.capacity = 0;
    recordWGSLKernelNodes(graph);
    const nodes = gather.getCommandNodes(graph);
    const emptyMask = nodes.find(node => node.id.endsWith('-empty-mask'))!;
    const binding = (node: GPUCommandNode, name: string) =>
      getWGSLKernelNodeRecord(graph, node.id)!.bindings.find(binding => binding.name === name)!
        .view;
    executeCommandNodesOnCPU([emptyMask], memory);
    expect(Array.from(memory.readUint32(binding(emptyMask, 'maskOut')))).toEqual([1]);
    const rows = nodes.find(node => node.id.endsWith('-rows'))!;
    const publish = nodes.find(node => node.id.endsWith('-publish'))!;
    // Published compaction's hand-worked result for mask [1]: unmatched row [0], count 1.
    // Execute the owned row/publish shaders here; the headless case runs the complete graph.
    memory.write(binding(rows, 'matchedIn'), Uint32Array.of(0));
    memory.write(binding(rows, 'unmatchedIds'), Uint32Array.of(0));
    memory.write(binding(rows, 'unmatchedCount'), Uint32Array.of(1));
    executeCommandNodesOnCPU([rows, publish], memory);
    expect(Array.from(memory.readUint32(output.rowCount))).toEqual([1]);
    expect(Array.from(memory.readUint32(output.leftRows))).toEqual([0]);
    expect(Array.from(memory.readUint32(output.rightRows))).toEqual([0xffffffff]);
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});

it.each([
  'neighbors',
  'pairs'
] as const)('G1 captures declared %s maps once before graph declaration', inputKind => {
  const {device, graph, view, scalar} = fixture();
  const reads = new Map<string, number>();
  const getters = <Props extends object>(label: string, fields: Props): Props =>
    Object.defineProperties(
      {},
      Object.fromEntries(
        Object.entries(fields).map(([name, value]) => [
          name,
          {
            enumerable: true,
            get() {
              const path = `${label}.${name}`;
              reads.set(path, (reads.get(path) ?? 0) + 1);
              if (reads.get(path)! > 1) throw new Error(`read twice: ${path}`);
              return value;
            }
          }
        ])
      )
    ) as Props;
  const rows = view('uint32'),
    counts = view('uint32'),
    source = view('float32');
  const columns = [getters('column', {source, output: view('float32')})];
  const output = getters('output', {
    leftRows: view('uint32'),
    rightRows: view('uint32'),
    rowCount: scalar(),
    status: getters('status', {
      overflow: scalar(),
      candidateOverflow: scalar(),
      matchedCount: scalar()
    })
  });
  const input =
    inputKind === 'neighbors'
      ? {neighbors: getters('neighbors', {rows, counts, capacity: 1})}
      : {
          pairs: getters('pairs', {
            leftRows: view('uint32'),
            rightRows: view('uint32'),
            status: getters('pairs.status', {
              count: scalar(),
              overflow: scalar(),
              requiredCount: scalar(),
              candidateOverflow: scalar()
            })
          }),
          unmatchedLeft: getters('unmatchedLeft', {
            ids: view('uint32'),
            status: getters('unmatchedLeft.status', {
              count: scalar(),
              overflow: scalar(),
              requiredCount: scalar(),
              candidateOverflow: scalar()
            })
          })
        };
  const props = getters('props', {
    id: 'getter-gather',
    how: 'left' as const,
    ...input,
    leftColumns: columns,
    rightColumns: [getters('rightColumn', {source, output: view('float32')})],
    slotColumns: [getters('slotColumn', {source, output: view('float32')})],
    overflowSources: [scalar()],
    candidateOverflowSources: [scalar()],
    output
  });
  try {
    const gather = new GPUPairGather(props);
    columns.length = 0;
    const nodes = gather.getCommandNodes(graph);
    expect(
      nodes.some(node =>
        node.resources?.some(resource => 'buffer' in resource && resource.buffer === source)
      )
    ).toBe(true);
    expect([...reads.values()].every(count => count === 1)).toBe(true);
    expect(gather.props.leftColumns).toHaveLength(1);
  } finally {
    device.destroy();
  }
});
