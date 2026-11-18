// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {
  decodeGPUGeometryValidityIssueCode,
  GPU_GEOMETRY_VALIDITY_BIT
} from '../../../src/gpu-spatial-analysis/geometry-validity/geometry-validity-types';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getUsablePolygonCount} from '../../../../../examples/deck/spatial-analysis-explorer/modes/validity-summary';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('publishes a nonzero usable flag separately from point issue codes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const positions = createTransientView(graph, 'positions', 'float32x2', 2);
  const issueCode = createTransientView(graph, 'issues', 'uint32', 2);
  const validity = createTransientView(graph, 'validity', 'uint32', 2);
  const nodes = new GPUGeometryValidity({
    points: {kind: 'points', positions},
    output: {issueCode, validity}
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(positions, Float32Array.of(1, 2, Number.NaN, 3));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(issueCode))).toEqual([0, 1]);
  expect(Array.from(memory.readUint32(validity))).toEqual([1, 0]);
  expect(
    () =>
      new GPUGeometryValidity({
        points: {kind: 'points', positions},
        output: {issueCode, validity: issueCode}
      })
  ).toThrow(/outputs must not share buffers with inputs/);
});

it('rejects issueCode aliases with polygon intersection status', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const words = (length: number) =>
    createTransientView(graph, `alias-${serial++}`, 'uint32', length);
  const issueCode = words(1);
  expect(
    () =>
      new GPUGeometryValidity({
        polygons: {
          kind: 'polygons',
          positions: createTransientView(graph, 'triangle', 'float32x2', 3),
          featureOffsets: words(2),
          polygonOffsets: words(2),
          ringOffsets: words(2)
        },
        intersectionCapacity: 4,
        output: {issueCode, validity: words(1), status: {overflow: issueCode}}
      })
  ).toThrow(/issueCode.*overflow|overflow.*issueCode/);
});

it.each([
  'points',
  'lines'
] as const)('clears supplied intersection scalars for finite %s on every submission', kind => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const positions = createTransientView(graph, 'simple-input', 'float32x2', 2);
  const offsets = createTransientView(graph, 'simple-offsets', 'uint32', 2);
  const count = kind === 'points' ? 2 : 1;
  const issueCode = createTransientView(graph, 'simple-issues', 'uint32', count);
  const validity = createTransientView(graph, 'simple-validity', 'uint32', count);
  const overflow = createTransientView(graph, 'simple-overflow', 'uint32', 1);
  const intersectionCount = createTransientView(graph, 'simple-intersections', 'uint32', 1);
  const geometry =
    kind === 'points'
      ? {points: {kind, positions}}
      : {lines: {kind, positions, lineOffsets: offsets}};
  const nodes = new GPUGeometryValidity({
    ...geometry,
    output: {issueCode, validity, status: {overflow, intersectionCount}}
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(positions, Float32Array.of(0, 0, 2, 0));
  memory.write(offsets, Uint32Array.of(0, 2));
  for (let submission = 0; submission < 2; submission++) {
    memory.write(overflow, Uint32Array.of(1));
    memory.write(intersectionCount, Uint32Array.of(7));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(validity))).toEqual(Array(count).fill(1));
    expect(Array.from(memory.readUint32(overflow))).toEqual([0]);
    expect(Array.from(memory.readUint32(intersectionCount))).toEqual([0]);
  }
});

it('keeps advisory polygon codes usable and suppresses usability after incomplete checks', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
    createTransientView(graph, `polygon-validity-${serial++}`, format, length);
  const issueCode = view('uint32', 4);
  const validity = view('uint32', 4);
  const overflow = view('uint32', 1);
  const nodes = new GPUGeometryValidity({
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 12),
      featureOffsets: view('uint32', 5),
      polygonOffsets: view('uint32', 5),
      ringOffsets: view('uint32', 5)
    },
    intersectionCapacity: 4,
    output: {issueCode, validity, status: {overflow}}
  })
    .getCommandNodes(graph)
    .filter(node => node.id.endsWith('-validity'));
  expect(nodes).toHaveLength(1);
  const memory = createCPUGraphMemory();
  const bits = GPU_GEOMETRY_VALIDITY_BIT;
  memory.write(
    issueCode,
    Uint32Array.of(bits.badOrientation, bits.repeatedVertex, bits.selfIntersection, bits.uncertain)
  );
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(validity))).toEqual([1, 1, 0, 0]);
  memory.write(overflow, Uint32Array.of(1));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(validity))).toEqual([0, 0, 0, 0]);
  const decoded = decodeGPUGeometryValidityIssueCode(bits.badOrientation | bits.repeatedVertex);
  expect(decoded.badOrientation).toBe(true);
  expect(decoded.repeatedVertex).toBe(true);
  expect(decoded.nonFinite).toBe(false);
  expect(decoded.hasStructuralIssue).toBe(false);
});

it('decodes only observed structural issues instead of certifying incomplete checks', () => {
  const decoded = decodeGPUGeometryValidityIssueCode(0);
  expect(decoded).toHaveProperty('hasStructuralIssue', false);
  expect(decoded).not.toHaveProperty('structurallyValid');
});

it('the explorer counts published usability and suppresses incomplete runs', () => {
  // Codes [0,128,8,16] have published usability [1,1,1,0] on a complete run.
  const validity = Uint32Array.of(1, 1, 1, 0);
  expect(getUsablePolygonCount(validity, 0)).toBe(3);
  expect(getUsablePolygonCount(validity, 1)).toBeUndefined();
});
