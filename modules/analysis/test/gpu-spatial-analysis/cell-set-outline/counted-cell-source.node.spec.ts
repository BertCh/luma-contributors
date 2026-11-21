// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import type {
  GPUCellSetOutlineInput,
  GPUCellCompactionInput,
  GPUCellSetOutlineProps,
  GPUCellCompactionProps
} from '../../../src/gpu-spatial-analysis';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline';
import {GPUCellCompaction} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-compaction';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const producer of ['outline', 'compact', 'uncompact'] as const) {
  for (const reportLoss of [false, true]) {
    it(`carries counted cell source loss independently of destination sizing (${producer}, report ${reportLoss})`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const scalar = () => createTransientView(graph, `scalar-${serial++}`, 'uint32', 1);
      const source = {
        cells: createTransientView(graph, 'source', 'uint32x2', 4),
        status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
      };
      const status = {
        count: scalar(),
        overflow: scalar(),
        requiredCount: scalar(),
        candidateOverflow: reportLoss ? scalar() : undefined
      };
      const cells = createTransientView(graph, 'destination', 'uint32x2', 8);
      const contributor =
        producer === 'outline'
          ? new GPUCellSetOutline({
              family: 'quadbin',
              source,
              output: {
                cells,
                rows: createTransientView(graph, 'rows', 'uint32', 8),
                edgeIndices: createTransientView(graph, 'edges', 'uint32', 8),
                endpoints: createTransientView(graph, 'endpoints', 'float32x4', 8),
                status
              }
            })
          : new GPUCellCompaction({
              family: 'quadbin',
              source,
              operation: {type: producer, resolution: 2},
              output: {cells, status}
            });
      const original = {cells: source.cells, status: {...source.status}};
      source.cells = createTransientView(graph, 'replacement-source', 'uint32x2', 4);
      source.status.count = scalar();
      source.status.overflow = scalar();
      source.status.candidateOverflow = scalar();
      source.status = {count: scalar(), overflow: scalar(), candidateOverflow: scalar()};
      expect(contributor.props.cells).toBe(original.cells);
      expect(contributor.props.count).toBe(original.status.count);
      expect(contributor.props.source?.status.overflow).toBe(original.status.overflow);
      expect(contributor.props.source?.status.candidateOverflow).toBe(
        original.status.candidateOverflow
      );
      const nodes = contributor
        .getCommandNodes(graph)
        .filter(node => node.id.endsWith('-source-loss'));
      expect(nodes).toHaveLength(1);
      const memory = createCPUGraphMemory();
      try {
        // A missing sibling cannot be recovered downstream. Local capacity and required
        // size still describe the retained source; neither is an inherited-loss flag.
        for (const localOverflow of [0, 1]) {
          for (const [overflow, candidateOverflow] of [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0]
          ]) {
            memory.write(original.status.overflow, Uint32Array.of(overflow));
            memory.write(original.status.candidateOverflow, Uint32Array.of(candidateOverflow));
            memory.write(status.count, Uint32Array.of(4));
            memory.write(status.overflow, Uint32Array.of(localOverflow));
            memory.write(status.requiredCount, Uint32Array.of(6));
            executeCommandNodesOnCPU(nodes, memory);
            const loss = overflow || candidateOverflow;
            expect(memory.readUint32(status.count)[0]).toBe(!reportLoss && loss ? 0 : 4);
            expect(memory.readUint32(status.overflow)[0]).toBe(localOverflow);
            expect(memory.readUint32(status.requiredCount)[0]).toBe(6);
            if (status.candidateOverflow)
              expect(memory.readUint32(status.candidateOverflow)[0]).toBe(loss ? 1 : 0);
          }
        }
      } finally {
        device.destroy();
      }
    });
  }
}

it('S2 exports both constructor declaration types while keeping legacy cells required', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `typed-input-${serial++}`, format, length);
  const source = {
    cells: view('uint32x2', 1),
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  };
  const status = {count: view('uint32', 1), overflow: view('uint32', 1)};
  const outlineInput: GPUCellSetOutlineInput = {
    family: 'quadbin',
    source,
    output: {
      rows: view('uint32', 4),
      cells: view('uint32x2', 4),
      edgeIndices: view('uint32', 4),
      endpoints: view('float32x4', 4),
      status
    }
  };
  const compactionInput: GPUCellCompactionInput = {
    family: 'quadbin',
    source,
    operation: {type: 'compact', resolution: 2},
    output: {cells: view('uint32x2', 4), status}
  };
  try {
    expect(new GPUCellSetOutline(outlineInput).props.cells).toBe(source.cells);
    expect(new GPUCellCompaction(compactionInput).props.cells).toBe(source.cells);
  } finally {
    device.destroy();
  }

  expectTypeOf<
    ConstructorParameters<typeof GPUCellSetOutline>[0]
  >().toEqualTypeOf<GPUCellSetOutlineInput>();
  expectTypeOf<
    ConstructorParameters<typeof GPUCellCompaction>[0]
  >().toEqualTypeOf<GPUCellCompactionInput>();
  expectTypeOf<GPUCellSetOutlineProps['cells']>().toEqualTypeOf<GraphDataView<'uint32x2'>>();
  expectTypeOf<GPUCellCompactionProps['cells']>().toEqualTypeOf<GraphDataView<'uint32x2'>>();
});
