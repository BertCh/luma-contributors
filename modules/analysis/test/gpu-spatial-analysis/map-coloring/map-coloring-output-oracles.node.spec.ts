// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMapColoringOnCPU} from '../../../src/gpu-spatial-analysis/map-coloring/cpu-map-coloring-operation';
import {CPUMapColoringOperation} from '../../../src/gpu-spatial-analysis/map-coloring/cpu-map-coloring-operation';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {
  GPUMapColoringOperation,
  type MapColoringOperationProps
} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {GPUMapColoring} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('direct contributor retains validated input/output descriptors before deferred graph declaration', () => {
  const device = createNullWebGPUDevice();
  const fixture = createOracleFixture(device);
  recordWGSLKernelNodes(fixture.graph);
  try {
    const colors = fixture.output('uint32', 4).view,
      converged = fixture.output('uint32', 1).view;
    const weights = {
      offsets: fixture.input('uint32', [0, 0, 0, 0, 0]),
      neighbors: fixture.input('uint32', [0]),
      weights: fixture.input('float32', [1])
    };
    const props = {
      weights,
      parameters: fixture.input('uint32', [19]),
      maximumIterations: 1,
      output: {colors, status: {converged}}
    };
    const contributor = new GPUMapColoring(props);
    props.output.colors = fixture.output('uint32', 1).view;
    props.output.status.converged = fixture.output('uint32', 1).view;
    props.weights.offsets = fixture.input('uint32', [0, 0]);
    const nodes = contributor.getCommandNodes(fixture.graph);
    const bindings = nodes.flatMap(
      node => getWGSLKernelNodeRecord(fixture.graph, node.id)?.bindings ?? []
    );
    expect(bindings.some(binding => binding.view === colors)).toBe(true);
    expect(bindings.some(binding => binding.view === converged)).toBe(true);
    expect(bindings.some(binding => binding.view === props.output.colors)).toBe(false);
    expect(
      'offsets' in contributor.props.weights ? contributor.props.weights.offsets.length : -1
    ).toBe(5);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

const makeHubInputs = () => {
  const vector = (values: Uint32Array | Float32Array): GPUCPUVector => ({
    format: values instanceof Float32Array ? 'float32' : 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  return {
    offsets: vector(Uint32Array.of(0, 1, 2, 3, 6)),
    neighbors: vector(Uint32Array.of(3, 3, 3, 0, 1, 2)),
    weights: vector(Float32Array.of(1, 1, 1, 1, 1, 1)),
    parameters: vector(Uint32Array.of(19))
  };
};
const makeOperationProps = (): MapColoringOperationProps => ({
  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  parameters: 'parameters',
  rowCount: 4,
  maximumIterations: 1,
  output: {colors: 'colors', status: {converged: 'done', iterationCount: 'rounds'}}
});

it.each([
  ['CPU', CPUMapColoringOperation],
  ['GPU', GPUMapColoringOperation]
] as const)('%s coloring operation captures an invalid budget getter once', (_label, Operation) => {
  const props = makeOperationProps();
  let reads = 0;
  Object.defineProperty(props, 'maximumIterations', {get: () => (reads++ === 0 ? 0 : 1)});
  expect(() => new Operation(props)).toThrow(/maximumIterations/);
  expect(reads).toBe(1);
});

it('CPU operation captures a valid one-read budget getter before execution', async () => {
  const props = makeOperationProps();
  let reads = 0;
  Object.defineProperty(props, 'maximumIterations', {get: () => (++reads === 1 ? 1 : 2)});
  const operation = new CPUMapColoringOperation(props);
  const result = await operation.execute({inputs: makeHubInputs()});
  expect(reads).toBe(1);
  expect(Array.from(result['colors'] as Uint32Array)).toEqual([
    0xffffffff, 0xffffffff, 0xffffffff, 0
  ]);
  expect(Array.from(result['done'] as Uint32Array)).toEqual([0]);
  expect(Array.from(result['rounds'] as Uint32Array)).toEqual([1]);
});

it('asymmetric adjacency distinguishes the unbudgeted priority sweep from the bounded row sweep', () => {
  // Row 1 has degree one and row 0 degree zero, so priority order visits 1 before 0.
  // With no reverse edge, row 0 does not see row 1. The bounded row sweep visits 0 first.
  for (const [budget, expected] of [
    [undefined, [0, 0]],
    [1, [0, 1]]
  ] as const) {
    const result = computeMapColoringOnCPU(
      Uint32Array.of(0, 0, 1),
      Uint32Array.of(0),
      Uint32Array.of(19),
      'largest-degree-first',
      budget
    );
    expect(Array.from(result.colors)).toEqual(expected);
    expect(Array.from(result.iterationCount)).toEqual([1]);
    expect(Array.from(result.converged)).toEqual([1]);
  }
});

it('CPU operation retains declared output names and one-round budget after caller mutation', async () => {
  const props = makeOperationProps();
  const operation = new CPUMapColoringOperation(props);
  props.maximumIterations = 2;
  props.output.status!.converged = 'other';
  const result = await operation.execute({inputs: makeHubInputs()});
  expect(Object.keys(result).sort()).toEqual(['colors', 'done', 'rounds']);
  expect(Array.from(result['colors'] as Uint32Array)).toEqual([
    0xffffffff, 0xffffffff, 0xffffffff, 0
  ]);
  expect(Array.from(result['done'] as Uint32Array)).toEqual([0]);
});

it('GPU operation lowering retains its one-round budget and captured status ports', () => {
  const props = makeOperationProps();
  const operation = new GPUMapColoringOperation(props);
  props.maximumIterations = 2;
  props.output.status!.converged = 'other';
  const contributor = constructNetworkOperationContributor(operation, {
    offsets: {format: 'uint32', length: 5},
    neighbors: {format: 'uint32', length: 6},
    weights: {format: 'float32', length: 6},
    parameters: {format: 'uint32', length: 1}
  }) as unknown as {maximumIterations: number; props: {output: {status?: {converged?: unknown}}}};
  expect(contributor.maximumIterations).toBe(1);
  expect(contributor.props.output.status?.converged).toBeDefined();
});

it('CPU operation honors one round on a hub-last star and reports remaining uncolored rows', async () => {
  const operation = new CPUMapColoringOperation({
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    parameters: 'parameters',
    rowCount: 4,
    maximumIterations: 1,
    output: {
      colors: 'colors',
      colorCount: 'colorCount',
      conflictCount: 'conflictCount',
      status: {converged: 'converged', iterationCount: 'iterationCount'}
    }
  });
  const vector = (values: Uint32Array | Float32Array): GPUCPUVector => ({
    format: values instanceof Float32Array ? 'float32' : 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const result = await operation.execute({
    inputs: {
      offsets: vector(Uint32Array.of(0, 1, 2, 3, 6)),
      neighbors: vector(Uint32Array.of(3, 3, 3, 0, 1, 2)),
      weights: vector(Float32Array.of(1, 1, 1, 1, 1, 1)),
      parameters: vector(Uint32Array.of(19))
    }
  });
  // Degree 3 puts the last hub first in priority. In the defined CPU ascending-row sweep,
  // all leaves visit it before it is colored, so the single round colors only the hub.
  expect(Array.from(result['colors'] as Uint32Array)).toEqual([
    0xffffffff, 0xffffffff, 0xffffffff, 0
  ]);
  expect(Array.from(result['colorCount'] as Uint32Array)).toEqual([1]);
  expect(Array.from(result['converged'] as Uint32Array)).toEqual([0]);
  expect(Array.from(result['iterationCount'] as Uint32Array)).toEqual([1]);
  expect(Array.from(result['conflictCount'] as Uint32Array)).toEqual([0]);
});

it('CPU coloring gives every isolate color zero and finishes one sweep', () => {
  // No row has a neighbor, so every row independently selects the first color, zero.
  for (const priority of ['random', 'largest-degree-first'] as const) {
    const result = computeMapColoringOnCPU(
      Uint32Array.of(0, 0, 0, 0, 0),
      Uint32Array.of(0),
      Uint32Array.of(19),
      priority
    );
    expect(Array.from(result.colors)).toEqual([0, 0, 0, 0]);
    expect(Array.from(result.colorCount)).toEqual([1]);
    expect(Array.from(result.iterationCount)).toEqual([1]);
    expect(Array.from(result.conflictCount)).toEqual([0]);
    expect(Array.from(result.converged)).toEqual([1]);
  }
  // In a three-leaf star the hub's degree 3 outranks every degree-1 leaf, for any seed.
  // The hub chooses zero; each mutually nonadjacent leaf must then choose one.
  const star = computeMapColoringOnCPU(
    Uint32Array.of(0, 3, 4, 5, 6),
    Uint32Array.of(1, 2, 3, 0, 0, 0),
    Uint32Array.of(19),
    'largest-degree-first'
  );
  expect(Array.from(star.colors)).toEqual([0, 1, 1, 1]);
  expect(Array.from(star.colorCount)).toEqual([2]);
});

it('bounded CPU rounds converge after a second sweep and never count uncolored conflicts', () => {
  const result = computeMapColoringOnCPU(
    Uint32Array.of(0, 1, 2, 3, 6),
    Uint32Array.of(3, 3, 3, 0, 1, 2),
    Uint32Array.of(19),
    'largest-degree-first',
    2
  );
  expect(Array.from(result.colors)).toEqual([1, 1, 1, 0]);
  expect(Array.from(result.iterationCount)).toEqual([2]);
  expect(Array.from(result.converged)).toEqual([1]);
  // Two mutually adjacent leaves both wait for the higher-degree, last-row hub. Their
  // uncolored sentinels are not colors and must not create a reported conflict.
  const partial = computeMapColoringOnCPU(
    Uint32Array.of(0, 2, 4, 5, 8),
    Uint32Array.of(1, 3, 0, 3, 3, 0, 1, 2),
    Uint32Array.of(19),
    'largest-degree-first',
    1
  );
  expect(Array.from(partial.colors)).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 0]);
  expect(Array.from(partial.conflictCount)).toEqual([0]);
});
