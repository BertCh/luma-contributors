// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellGeometry} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry';
import {GPUCellMeasures} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures';
import {GPUCellTopology} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-topology';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {describeGPUCommandGraph} from '../../../src/computation-record/index';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const producer of ['geometry', 'measures', 'measures-fp64', 'topology'] as const) {
  it(`LENS-104 records fresh ${producer} chunk declarations and deduplicates their source`, () => {
    const device = createNullWebGPUDevice();
    Object.defineProperty(device, 'props', {
      value: {...device.props, _cacheShaders: false, _cachePipelines: false}
    });
    Object.defineProperty(device, 'createComputePipeline', {
      value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy: () => {}})
    });
    try {
      for (const recording of [false, true]) {
        const graph = new GPUCommandGraph(device, {id: `${producer}-${recording}`});
        if (recording) recordWGSLKernelNodes(graph);
        const cells = createVectorView(
          'cells',
          'uint32x2',
          [1, 257].map((length, chunk) =>
            createTransientView(graph, `cells-${chunk}`, 'uint32x2', length)
          )
        );
        const centers = createVectorView(
          'centers',
          'float32x2',
          [1, 257].map((length, chunk) =>
            createTransientView(graph, `centers-${chunk}`, 'float32x2', length)
          )
        );
        const areas = createVectorView(
          'areas',
          'float32',
          [1, 257].map((length, chunk) =>
            createTransientView(graph, `areas-${chunk}`, 'float32', length)
          )
        );
        const parents = createVectorView(
          'parents',
          'uint32x2',
          [1, 257].map((length, chunk) =>
            createTransientView(graph, `parents-${chunk}`, 'uint32x2', length)
          )
        );
        const contributor =
          producer === 'geometry'
            ? new GPUCellGeometry({id: producer, family: 's2', cells, output: {centers}})
            : producer === 'topology'
              ? new GPUCellTopology({
                  id: producer,
                  family: 's2',
                  cells,
                  operation: {type: 'parent', resolution: 0},
                  output: {cells: parents}
                })
              : new GPUCellMeasures({
                  id: producer,
                  family: 's2',
                  cells,
                  precision: producer === 'measures-fp64' ? 'double-single' : 'float32',
                  output: {areas}
                });
        const nodes = contributor.getCommandNodes(graph);
        const compute = nodes.filter(node => node.type === 'compute');
        expect(compute).toHaveLength(2);
        for (const node of nodes) graph.add(node);
        const records = compute.map(node => getWGSLKernelNodeRecord(graph, node.id));
        if (recording) {
          for (const [chunk, record] of records.entries()) {
            expect(record, compute[chunk]!.id).toBeDefined();
            expect(record).toMatchObject({
              id: compute[chunk]!.id,
              entryPoint: 'main',
              constants: {},
              textures: [],
              workgroupSize: [256, 1, 1],
              workgroups: [chunk === 0 ? 1 : 2, 1, 1],
              invocationCount: chunk === 0 ? 1 : 257,
              indirect: false
            });
            expect(record!.bindings.find(binding => binding.name === 'cells')!.view).toBe(
              cells.data[chunk]
            );
            expect(record!.bindings.some(binding => binding.access === 'read_write')).toBe(true);
            expect(record!.source).toContain('fn main(');
            expect(record!.source).toContain(record!.body);
          }
          expect(records[0]!.source).toBe(records[1]!.source);
          expect(records[0]!.bindings[0]!.view).not.toBe(records[1]!.bindings[0]!.view);
          if (producer === 'measures-fp64') expect(records[0]!.source).toContain('fn sum_fp64');
        } else {
          expect(records).toEqual([undefined, undefined]);
        }
        const compiled = graph.compile();
        try {
          const described = describeGPUCommandGraph(compiled, {graph, sourceText: true});
          const describedCompute = described.graph.nodes.filter(node =>
            compute.some(kernel => kernel.id === node.id)
          );
          if (recording) {
            expect(described.sources).toHaveLength(1);
            for (const node of describedCompute) {
              expect(node.source).toBe(described.sources[0]!.id);
              expect(node.bindings!.find(binding => binding.name === 'cells')!.resource).toBe(
                `cells-${node.id.endsWith('-0') ? 0 : 1}`
              );
            }
            expect(described.sources[0]!.text).toBe(records[0]!.source);
            // Inspect separately only after description: standalone compilation must not be what
            // supplies the selected graph's provenance. This also checks fp64 assembly.
            for (const [chunk, record] of records.entries()) {
              expect(getGPUCommandNodeKernelSources([compute[chunk]!])[0]!.source).toBe(
                record!.source
              );
            }
          } else {
            expect(described.sources).toEqual([]);
            expect(describedCompute.every(node => node.source === undefined)).toBe(true);
          }
        } finally {
          compiled.destroy();
        }
      }
    } finally {
      device.destroy();
    }
  });
}
