// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover';
import {GPUCellCompaction} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-compaction';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const family of ['quadbin', 'h3'] as const) {
  for (const diagnostic of [false, true]) {
    for (const publishComplete of [false, true]) {
      for (const reportLoss of [false, true]) {
        it(`B1 separates coverage and capacity loss (${family}, diagnostic ${diagnostic}, complete ${publishComplete}, consumer loss ${reportLoss})`, () => {
          const device = createNullWebGPUDevice();
          try {
            const graph = new GPUCommandGraph(device);
            recordWGSLKernelNodes(graph);
            let serial = 0;
            const scalar = (length = 1) =>
              createTransientView(graph, `scalar-${serial++}`, 'uint32', length);
            const positions = createTransientView(graph, 'positions', 'float32x2', 4);
            const featureOffsets = scalar(2),
              polygonOffsets = scalar(2),
              ringOffsets = scalar(2);
            const source = {
              cells: createTransientView(graph, 'cells', 'uint32x2', 4),
              status: {
                count: scalar(),
                overflow: scalar(),
                candidateOverflow: scalar(),
                complete: publishComplete ? scalar() : undefined,
                uncertainCount: diagnostic ? scalar() : undefined
              }
            };
            const nodes = new GPUCellCover({
              family,
              resolution: 0,
              positions,
              featureOffsets,
              polygonOffsets,
              ringOffsets,
              candidateCapacity: 4,
              candidateEnumeration: 'bounds',
              edgeSlabs: false,
              output: {...source, featureIds: scalar(4)}
            }).getCommandNodes(graph);
            const finalize = getWGSLKernelNodeRecord(graph, 'cell-cover-finalize')!;
            const binding = (name: string) =>
              finalize.bindings.find(entry => entry.name === name)!.view as GraphDataView<'uint32'>;
            const status = {
              count: scalar(),
              candidateOverflow: reportLoss ? scalar() : undefined,
              overflow: scalar()
            };
            const consumer = new GPUCellCompaction({
              family,
              source,
              operation: {type: 'compact', resolution: 0},
              output: {cells: createTransientView(graph, 'compacted', 'uint32x2', 4), status}
            })
              .getCommandNodes(graph)
              .filter(node => node.id.endsWith('-source-loss'));
            const memory = createCPUGraphMemory();
            memory.write(featureOffsets, Uint32Array.of(0, 1));
            memory.write(polygonOffsets, Uint32Array.of(0, 1));
            memory.write(ringOffsets, Uint32Array.of(0, 4));
            // One feature: north=90 exceeds both latitude limits, north=80 does not.
            // Inject only the independent candidate-scan/accept-scan results to isolate status
            // publication. Candidate demand 5 exceeds capacity 4, while demand 4 fits exactly.
            for (const [clipped, capacityLoss, rejected] of [
              [1, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [0, 0, 0],
              [0, 0, 1],
              [0, 0, 0]
            ]) {
              memory.write(
                positions,
                rejected
                  ? Float32Array.of(175, 10, -175, 10, -175, 80, 175, 80)
                  : Float32Array.of(-10, 10, 10, 10, 10, clipped ? 90 : 80, -10, clipped ? 90 : 80)
              );
              memory.write(source.status.count, Uint32Array.of(99));
              memory.write(source.status.overflow, Uint32Array.of(99));
              memory.write(source.status.candidateOverflow, Uint32Array.of(99));
              if (source.status.complete) memory.write(source.status.complete, Uint32Array.of(99));
              executeCommandNodesOnCPU(
                nodes.filter(node => /-clear-/.test(node.id) || node.id.endsWith('-count')),
                memory
              );
              memory.write(binding('starts'), Uint32Array.of(0, capacityLoss ? 5 : 4));
              memory.write(binding('accepted'), Uint32Array.of(0, 0, 0, rejected ? 0 : 2));
              executeCommandNodesOnCPU(
                nodes.filter(node => /-(finalize|publish|completeness|diagnostics)$/.test(node.id)),
                memory
              );
              const coverageLoss = clipped || rejected;
              const readable = !publishComplete && coverageLoss ? 0 : rejected ? 0 : 2;
              expect(memory.readUint32(source.status.count)[0]).toBe(readable);
              expect(memory.readUint32(source.status.candidateOverflow)[0]).toBe(capacityLoss);
              // S1: this word is poisoned above and the actual cover publisher executes.
              expect(memory.readUint32(source.status.overflow)[0]).toBe(0);
              if (source.status.complete)
                expect(memory.readUint32(source.status.complete)[0]).toBe(
                  coverageLoss || capacityLoss ? 0 : 1
                );
              if (source.status.uncertainCount)
                expect(memory.readUint32(source.status.uncertainCount)[0]).toBe(clipped);
              memory.write(status.count, Uint32Array.of(readable));
              executeCommandNodesOnCPU(consumer, memory);
              const inherited = capacityLoss || (publishComplete && coverageLoss);
              expect(memory.readUint32(status.count)[0]).toBe(
                !reportLoss && inherited ? 0 : readable
              );
              if (status.candidateOverflow)
                expect(memory.readUint32(status.candidateOverflow)[0]).toBe(inherited ? 1 : 0);
              // Consumer local overflow is asserted by the full GPU graph, not by unrun writers.
            }
          } finally {
            device.destroy();
          }
        });
      }
    }
  }
}

it('B2 keeps shared-buffer public status words out of mixed storage usage scopes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const storage = createTransientView(graph, 'status-storage', 'uint32', 256);
    const scalar = (offset: number) =>
      graph.createDataView(storage.buffer, {format: 'uint32', byteOffset: offset, length: 1});
    const status = {
      candidateOverflow: scalar(0),
      uncertainCount: scalar(256),
      candidateCount: scalar(512),
      complete: scalar(768),
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1)
    };
    const contributor = new GPUCellCover({
      family: 'quadbin',
      resolution: 0,
      candidateCapacity: 4,
      candidateEnumeration: 'bounds',
      edgeSlabs: false,
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      output: {
        cells: createTransientView(graph, 'cells', 'uint32x2', 4),
        featureIds: createTransientView(graph, 'ids', 'uint32', 4),
        status
      }
    });
    expect(() => new GPUCellCover({...contributor.props, family: 's2'})).toThrow(
      /complete.*H3\/Quadbin/
    );
    const nodes = contributor.getCommandNodes(graph);
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      if (!record) continue;
      for (const read of record.bindings.filter(entry => entry.access === 'read'))
        for (const write of record.bindings.filter(entry => entry.access === 'read_write'))
          expect(read.view.buffer, node.id).not.toBe(write.view.buffer);
    }
  } finally {
    device.destroy();
  }
});
