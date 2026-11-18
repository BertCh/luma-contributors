// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createTopologyOutput} from '../geometry-refinement/topology-output';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function fixture(
  makeValid = false,
  rounded = false,
  onlyNodedUncertainty = false,
  featureCount = 1
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `final1-${serial++}`, format, length);
  const output = createTopologyOutput(view);
  output.nodedSegments.status.uncertainCount = view('uint32', 1);
  output.directedEdges.status.uncertainCount = view('uint32', 1);
  output.status.uncertainCount = view('uint32', 1);
  if (onlyNodedUncertainty) {
    delete output.status.uncertainCount;
    delete output.directedEdges.status.uncertainCount;
  }
  const polygons = {
    kind: 'polygons' as const,
    positions: view('float32x2', 4 * featureCount),
    ringOffsets: view('uint32', featureCount + 1),
    polygonOffsets: view('uint32', featureCount + 1),
    featureOffsets: view('uint32', featureCount + 1),
    sourceIds: view('uint32', featureCount)
  };
  const common = {
    id: 'final1',
    output,
    intersectionCapacity: 4,
    ...(rounded ? {snapRounding: {gridSize: 1, lattice: view('float32', 4)}} : {}),
    precision: {
      vertexTolerance: 1e-5,
      predicates: 'exact-or-uncertain' as const,
      storage: 'float32' as const
    }
  };
  const nodes = (
    makeValid
      ? new GPUMakeValid({...common, polygons})
      : new GPUPolygonize({
          ...common,
          lines: {kind: 'lines', positions: polygons.positions, lineOffsets: polygons.ringOffsets}
        })
  ).getCommandNodes(graph);
  const record = (suffix: string) => getWGSLKernelNodeRecord(graph, `final1-${suffix}`)!;
  const binding = (suffix: string, name: string) =>
    record(suffix).bindings.find(value => value.name === name)!.view;
  return {graph, output, polygons, nodes, record, binding, memory: createCPUGraphMemory()};
}

it.each([
  false,
  true
])('noding keeps intermediate loss out of final capacity (rounded=%s)', rounded => {
  const {output, nodes, binding, memory} = fixture(false, rounded);
  if (rounded) {
    const scan = binding('noding-status', 'candidateScan');
    memory.write(scan, new Uint32Array(scan.length).fill(2));
    const state = binding('noding-status', 'state');
    const stateWords = new Uint32Array(state.length);
    stateWords[1] = 1;
    memory.write(state, stateWords);
    memory.write(binding('noding-status-candidates', 'crossingOverflow'), Uint32Array.of(1));
  } else {
    const scan = binding('noding-status', 'scan');
    memory.write(scan, new Uint32Array(scan.length).fill(1));
    memory.write(binding('noding-status', 'splitIncomplete'), Uint32Array.of(1));
    memory.write(output.nodedSegments.status.candidateOverflow, Uint32Array.of(1));
  }
  executeCommandNodesOnCPU(
    nodes.filter(node =>
      ['final1-noding-status-candidates', 'final1-noding-status'].includes(node.id)
    ),
    memory
  );
  expect(memory.readUint32(output.nodedSegments.status.requiredCount)[0]).toBe(1);
  expect(memory.readUint32(output.nodedSegments.status.candidateOverflow)[0]).toBe(1);
  expect(memory.readUint32(output.nodedSegments.status.overflow)[0]).toBe(0);
});

it('make-valid maps reordered retained feature rows to their original identities', () => {
  const {polygons, nodes, binding, memory, output} = fixture(true, false, false, 2);
  memory.write(polygons.featureOffsets, Uint32Array.of(0, 1, 2));
  memory.write(polygons.polygonOffsets, Uint32Array.of(0, 1, 2));
  memory.write(polygons.sourceIds, Uint32Array.of(7, 0xffffffff));
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === 'final1-ring-sources'),
    memory
  );
  memory.write(binding('polygon-layout', 'keepScan'), Uint32Array.of(1, 2, 2, 2, 2, 2, 2, 2));
  memory.write(binding('polygon-layout', 'keptIds'), Uint32Array.of(0, 1));
  memory.write(binding('polygon-layout', 'faceSourceIds'), Uint32Array.of(1, 0));
  executeCommandNodesOnCPU(
    nodes.filter(node => ['final1-polygon-layout', 'final1-publish-source-ids'].includes(node.id)),
    memory
  );
  expect(Array.from(memory.readUint32(output.polygons.sourceIds)).slice(0, 2)).toEqual([
    0xffffffff, 7
  ]);
});

it('snap-rounded noding propagates incidence scratch exhaustion independently of its pair list', () => {
  const {nodes, binding, memory, output} = fixture(false, true, false, 3);
  const scan = binding('noding-status', 'candidateScan');
  // Capacity is max(32 + 12 input vertices, 2 * (12 vertices + 4 crossings)) = 44.
  memory.write(
    scan,
    Uint32Array.from({length: scan.length}, (_, index) => Math.min(index + 1, 45))
  );
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === 'final1-noding-status-candidates'),
    memory
  );
  expect(memory.readUint32(output.nodedSegments.status.candidateOverflow)[0]).toBe(1);
});

it.each([
  {count: 32, required: 33, candidate: 0, overflow: 1},
  {count: 1, required: 1, candidate: 1, overflow: 0}
])('directed capacity and candidate loss have independent flags: %o', values => {
  const {output, nodes, memory} = fixture();
  memory.write(output.nodedSegments.status.count, Uint32Array.of(values.count));
  memory.write(output.nodedSegments.status.requiredCount, Uint32Array.of(values.required));
  memory.write(output.nodedSegments.status.candidateOverflow, Uint32Array.of(values.candidate));
  memory.write(output.nodedSegments.status.overflow, Uint32Array.of(values.overflow));
  executeCommandNodesOnCPU(
    nodes.filter(node =>
      [
        'final1-directed-edge-status',
        'final1-publish-status',
        'final1-publish-candidate-overflow'
      ].includes(node.id)
    ),
    memory
  );
  expect(Array.from(memory.readUint32(output.directedEdges.status.count))).toEqual([
    2 * values.count
  ]);
  expect(Array.from(memory.readUint32(output.directedEdges.status.requiredCount))).toEqual([
    2 * values.required
  ]);
  expect(Array.from(memory.readUint32(output.directedEdges.status.overflow))).toEqual([
    values.overflow
  ]);
  expect(Array.from(memory.readUint32(output.directedEdges.status.candidateOverflow))).toEqual([
    values.candidate
  ]);
  expect(Array.from(memory.readUint32(output.status.overflow))).toEqual([0]);
  expect(Array.from(memory.readUint32(output.status.candidateOverflow))).toEqual([1]);
});

it.each([
  false,
  true
])('writes every advertised uncertainty destination on a second submission (only noded=%s)', onlyNodedUncertainty => {
  const {graph, output, nodes, memory} = fixture(false, false, onlyNodedUncertainty);
  const destinations = [
    output.status.uncertainCount!,
    output.nodedSegments.status.uncertainCount!,
    output.directedEdges.status.uncertainCount!
  ].filter(value => value !== undefined);
  for (const expected of [3, 0]) {
    for (const destination of destinations) memory.write(destination, Uint32Array.of(777));
    const copies = nodes.filter(node => node.id.startsWith('final1-publish-uncertain-'));
    expect(copies).toHaveLength(destinations.length);
    // Replay the publisher after the split pass has produced its uncertainty count.
    const source = getWGSLKernelNodeRecord(graph, copies[0].id)!.bindings.find(
      value => value.name === 'uncertainCount'
    )!.view;
    memory.write(source, Uint32Array.of(expected));
    executeCommandNodesOnCPU(copies, memory);
    for (const destination of destinations)
      expect(memory.readUint32(destination)[0]).toBe(expected);
  }
});

it.each([
  0xfffffffe, 0xffffffff
])('make-valid uses feature rows for topology and preserves caller identity %i', identity => {
  const {polygons, nodes, binding, memory, output} = fixture(true);
  memory.write(polygons.featureOffsets, Uint32Array.of(0, 1));
  memory.write(polygons.polygonOffsets, Uint32Array.of(0, 1));
  memory.write(polygons.sourceIds, Uint32Array.of(identity));
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === 'final1-ring-sources'),
    memory
  );
  const groups = binding('polygonize-directed-edge-provenance', 'inputGroups');
  expect(memory.readUint32(groups)[0]).toBe(0);
  // Replay retained-face publication, with internal face row zero and one kept feature.
  memory.write(binding('polygon-layout', 'keepScan'), new Uint32Array(8).fill(1));
  memory.write(binding('polygon-layout', 'faceSourceIds'), new Uint32Array(8));
  executeCommandNodesOnCPU(
    nodes.filter(node =>
      ['final1-polygon-layout', 'final1-publish-count', 'final1-publish-source-ids'].includes(
        node.id
      )
    ),
    memory
  );
  expect(memory.readUint32(output.polygons.sourceIds)[0]).toBe(identity);
});
