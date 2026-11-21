// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {GPUTimeWindowFilter} from '../../src/dataframe-analysis/time-window-filter/gpu-time-window-filter';
import {getGPUTimeWindowFilterParameterValues} from '../../src/dataframe-analysis/time-window-filter/time-window-parameters';
import {GPUKeyLookup} from '../../src/dataframe-analysis/key-join/gpu-key-join';
import {GPUTemporalReduction} from '../../src/dataframe-analysis/temporal-reduction/gpu-temporal-reduction';
import {GPUSpatialPredicateJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

function fixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(format: Format, length = 1) =>
    createTransientView(graph, `final2-${serial++}`, format, length);
  const scalar = () => view('uint32');
  return {device, graph, view, scalar};
}

test.each([
  'rows',
  'input'
] as const)('final2 G5/G1 filter retains validated candidate status after mutation to %s', alias => {
  const {device, graph, view, scalar} = fixture();
  try {
    const timestamps = view('float32', 2),
      sourceIds = view('uint32', 2),
      window = view('float32', 8);
    const rows = scalar(),
      ids = scalar(),
      candidate = scalar();
    const status = {count: scalar(), overflow: scalar(), candidateOverflow: candidate};
    const contributor = new GPUTimeWindowFilter({
      sorted: true,
      timestamps,
      sourceIds,
      window,
      output: {rows, ids, status}
    });
    status.candidateOverflow =
      alias === 'rows'
        ? rows
        : graph.createDataView(sourceIds.buffer, {format: 'uint32', length: 1});
    const nodes = contributor.getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(timestamps, new Float32Array([0, 10]));
    memory.write(sourceIds, new Uint32Array([42, 99]));
    memory.write(window, getGPUTimeWindowFilterParameterValues({start: 10, end: 10}));
    memory.write(candidate, new Uint32Array([37]));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(rows))).toEqual([1]);
    expect(Array.from(memory.readUint32(ids))).toEqual([99]);
    expect(Array.from(memory.readUint32(candidate))).toEqual([0]);
    expect(Array.from(memory.readUint32(sourceIds))).toEqual([42, 99]);
  } finally {
    device.destroy();
  }
});

test.each([
  'join',
  'temporal'
] as const)('final2 G1 %s publisher retains the validated input and status maps', kind => {
  const {device, graph, view, scalar} = fixture();
  try {
    const input = scalar(),
      candidate = scalar();
    const status = {count: scalar(), overflow: scalar(), candidateOverflow: candidate};
    const compact = {ids: scalar(), status};
    const contributor =
      kind === 'join'
        ? new GPUKeyLookup({
            leftKeys: input,
            rightKeys: scalar(),
            compactMatchedLeftRows: true,
            output: {rows: compact}
          })
        : new GPUTemporalReduction({
            cellIds: input,
            timestamps: view('float32'),
            values: view('float32'),
            parameters: view('float32', 2),
            cellCount: 8,
            bucketCount: 1,
            output: {
              counts: view('uint32', 8),
              minimums: view('float32', 8),
              maximums: view('float32', 8),
              first: view('float32', 8),
              last: view('float32', 8),
              occupiedSlots: compact
            }
          });
    status.candidateOverflow = input;
    const publication = contributor
      .getCommandNodes(graph)
      .filter(node => node.id.endsWith('-publish'));
    expect(publication).toHaveLength(1);
    const memory = createCPUGraphMemory();
    memory.write(input, new Uint32Array([7]));
    memory.write(candidate, new Uint32Array([37]));
    executeCommandNodesOnCPU(publication, memory);
    expect(Array.from(memory.readUint32(input))).toEqual([7]);
    expect(Array.from(memory.readUint32(candidate))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

test('final2 G9 preserves non-enumerable requiredCount when completing weights status', () => {
  const {device, graph, view, scalar} = fixture();
  try {
    const requiredCount = scalar();
    const status = Object.defineProperty({count: scalar()}, 'requiredCount', {
      get: () => requiredCount
    });
    const join = new GPUSpatialPredicateJoin({
      left: {kind: 'points', positions: view('float32x2')},
      right: {kind: 'points', positions: view('float32x2')},
      predicate: 'intersects',
      output: {
        pairs: {
          leftRows: scalar(),
          rightRows: scalar(),
          status: {count: scalar(), overflow: scalar()}
        },
        weights: {offsets: view('uint32', 2), neighbors: scalar(), weights: view('float32'), status}
      }
    });
    const nodes = join
      .getCommandNodes(graph)
      .filter(node => /^spatial-predicate-join-finalize(?:-\d+)?$/.test(node.id));
    const memory = createCPUGraphMemory();
    memory.write(requiredCount, new Uint32Array([77]));
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id)!;
      const state = record.bindings.find(binding => binding.name === 'state')!.view;
      memory.write(state, new Uint32Array([1, 0, 1, 0, 0, 0, 0, 0]));
    }
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(requiredCount))).toEqual([1]);
  } finally {
    device.destroy();
  }
});
