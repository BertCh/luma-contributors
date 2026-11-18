// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createTopologyOutput} from '../geometry-refinement/topology-output';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('publishes feature count and ring requirement in separate cardinality ports', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `cardinality-${serial++}`, format, length);
  const base = createTopologyOutput(view);
  const requiredRingCount = view('uint32', 1);
  const output = {...base, status: {...base.status, requiredRingCount}};
  const nodes = new GPUPolygonize({
    id: 'cardinality',
    lines: {kind: 'lines', positions: view('float32x2', 10), lineOffsets: view('uint32', 3)},
    intersectionCapacity: 16,
    precision: {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'},
    output
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  // Assembly of two nested squares: two bounded features, four directed face walks,
  // and sufficient capacity. Replay the production publisher on that assembly result.
  memory.write(output.polygons.featureOffsets, Uint32Array.of(0, 1, 2, 2, 2, 2, 2, 2, 2));
  memory.write(
    getWGSLKernelNodeRecord(graph, 'cardinality-publish-status')!.bindings.find(
      binding => binding.name === 'ringRequiredCount'
    )!.view,
    Uint32Array.of(4)
  );
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === 'cardinality-publish-status'),
    memory
  );
  expect(memory.readUint32(output.status.count)[0]).toBe(2);
  expect(memory.readUint32(output.status.requiredRingCount)[0]).toBe(4);
  expect(memory.readUint32(output.status.overflow)[0]).toBe(0);
});

it.each([
  'polygonize',
  'make-valid'
] as const)('%s rejects empty and aliased ring requirements at construction', kind => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `status-review-${serial++}`, format, length);
  const output = createTopologyOutput(view);
  const build = () =>
    kind === 'polygonize'
      ? new GPUPolygonize({
          lines: {kind: 'lines', positions: view('float32x2', 4), lineOffsets: view('uint32', 2)},
          intersectionCapacity: 4,
          precision: {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'},
          output
        })
      : new GPUMakeValid({
          polygons: {
            kind: 'polygons',
            positions: view('float32x2', 4),
            featureOffsets: view('uint32', 2),
            polygonOffsets: view('uint32', 2),
            ringOffsets: view('uint32', 2)
          },
          intersectionCapacity: 4,
          precision: {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'},
          output
        });
  output.status.requiredRingCount = view('uint32', 0);
  expect(build).toThrow(/requiredRingCount/);
  output.status.requiredRingCount = output.status.count;
  expect(build).toThrow(/count.*requiredRingCount|requiredRingCount.*count/);
});
