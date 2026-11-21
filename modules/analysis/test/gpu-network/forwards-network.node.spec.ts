// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runInNewContext} from 'node:vm';
import {expect, it} from 'vitest';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {decodeGPUNetworkPathExtractionOutput} from '../../src/gpu-network/network-analysis/network-path-readback';
import {computeAlternativeRoutesOnCPU} from '../../src/gpu-network/alternative-routes/cpu-alternative-routes-operation';
import {decodeGPUAlternativeRoutes} from '../../src/gpu-network/alternative-routes/gpu-alternative-routes';
import {GPUNetworkCostMatrixOperation} from '../../src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation';
import {CPUNetworkTimeRoutingOperation} from '../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';
import {
  GPUNetworkTimeRoutingOperation,
  type NetworkTimeRoutingOperationProps
} from '../../src/gpu-network/network-optimization/gpu-network-time-routing-operation';
import * as visibility from '../../src/gpu-network/visibility-graph';
import {GPURadiationFlowModel} from '../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {constructNetworkOperationContributor} from '../utils/network-operation-test-utils';
import type {GPUNetworkTimeRouting} from '../../src/gpu-network/network-optimization/gpu-network-time-routing';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('B3 reads foreign realm path words and byte windows without narrowing IDs', () => {
  for (const expression of [
    'Uint32Array.of(257,2,3,4)',
    'Uint32Array.of(257,2,3,4).buffer',
    'new Uint8Array(Uint32Array.of(999,257,2,3,4).buffer,4,16)'
  ]) {
    const ids = runInNewContext(expression) as Uint32Array | Uint8Array | ArrayBuffer;
    const result = decodeGPUNetworkPathExtractionOutput({
      ids,
      pathFound: Uint32Array.of(1),
      lineOffsets: Uint32Array.of(0, 1),
      count: Uint32Array.of(1)
    });
    expect(Array.from(result[0].nodes)).toEqual([257]);
  }
  expect(() =>
    decodeGPUNetworkPathExtractionOutput({
      ids: new Uint8Array(3),
      pathFound: Uint32Array.of(1),
      lineOffsets: Uint32Array.of(0, 1),
      count: Uint32Array.of(1)
    })
  ).toThrow(/whole/);
});

it.each([
  0, 1
])('B2 decodes certified empty routes on a zero-node topology (%s queries)', queryCount => {
  const result = computeAlternativeRoutesOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0),
        neighbors: new Uint32Array(),
        weights: new Float32Array()
      },
      sources: new Uint32Array(queryCount),
      targets: new Uint32Array(queryCount)
    },
    {routeCount: 2}
  );
  // No vertices exist, so an invalid endpoint has no routes and the search is exhaustive.
  expect(decodeGPUAlternativeRoutes(result, {queryCount, routeCount: 2, nodeCount: 0})).toEqual({
    queries: queryCount ? [{count: 0, routes: [], complete: true}] : []
  });
});

it('N1 validates named matrix direction and noded evidence at construction', () => {
  const props = {
    nodeCount: 3,
    edgeCount: 2,
    rowCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    output: {costs: 'costs'}
  };
  expect(
    () => new GPUNetworkCostMatrixOperation({...props, direction: 'sideways' as never})
  ).toThrow(/direction.*start-to-target.*target-to-start/);
  const nodedTopology = {nodeCount: 'nodes', pieceCount: 'pieces', piecesOverflow: 'overflow'};
  expect(() => new GPUNetworkCostMatrixOperation({...props, nodedTopology})).toThrow(
    /nodedTopology.*sourceComplete/
  );
  for (const direction of [undefined, 'start-to-target', 'target-to-start'] as const)
    expect(() => new GPUNetworkCostMatrixOperation({...props, direction})).not.toThrow();
  expect(
    () =>
      new GPUNetworkCostMatrixOperation({
        ...props,
        nodedTopology,
        output: {costs: 'costs', status: {sourceComplete: 'complete'}}
      })
  ).not.toThrow();
});

const vector = (
  format: GPUCPUVector['format'],
  values: Uint32Array | Float32Array
): GPUCPUVector => {
  const length = values.length / (format.endsWith('x2') ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
};

it.each([
  [1, 0],
  [2 ** 25, 1.5]
])('elapsed named outputs preserve duration at clock %s', async (departure, duration) => {
  const props: NetworkTimeRoutingOperationProps = {
    stateCount: 2,
    transitionCount: 1,
    stateVertices: 'vertices',
    transitions: 'transitions',
    costs: 'costs',
    endpoints: 'endpoints',
    departure: 'departure',
    output: {
      arrivals: 'arrivals',
      predecessorTransitions: 'parents',
      summary: 'summary',
      ...{elapsedTimes: 'elapsed'}
    }
  };
  const gpu = new GPUNetworkTimeRoutingOperation(props);
  expect(gpu.outputs['elapsedTimes']).toMatchObject({
    value: 'elapsed',
    format: 'float32',
    length: 2
  });
  const contributor = constructNetworkOperationContributor(gpu, {
    stateVertices: {format: 'uint32', length: 2},
    transitions: {format: 'uint32x2', length: 1},
    costs: {format: 'float32x2', length: 1},
    endpoints: {format: 'uint32', length: 2},
    departure: {format: 'float32', length: 1}
  }) as GPUNetworkTimeRouting;
  expect(contributor.props.output.elapsedTimes).toMatchObject({format: 'float32', length: 2});
  const rows = await new CPUNetworkTimeRoutingOperation(props).execute({
    inputs: {
      stateVertices: vector('uint32', Uint32Array.of(0, 1)),
      transitions: vector('uint32x2', Uint32Array.of(0, 1)),
      costs: vector('float32x2', Float32Array.of(duration, 0)),
      endpoints: vector('uint32', Uint32Array.of(0, 1)),
      departure: vector('float32', Float32Array.of(departure))
    }
  });
  // An affine edge with zero slope takes exactly duration seconds, independent of the clock.
  expect(rows['elapsed']).toEqual(Float32Array.of(0, duration));
  expect(rows['arrivals']).toEqual(Float32Array.of(departure, departure + duration));
});

it('SCL-04 publishes the existing visibility planner as a direct top-level identity', () => {
  expect(visibility['GPU_NETWORK_VISIBILITY_GRAPH_PLANNER' as keyof typeof visibility]).toBe(
    visibility.addVisibilityGraphRecipe.executionPlanner
  );
});

it('S2 resolves and retains radiation normalization in the resident constructor', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2'>(format: Format, length: number) =>
    createTransientView(graph, `radiation-${serial++}`, format, length);
  try {
    const model = new GPURadiationFlowModel({
      positions: view('float32x2', 2),
      populations: view('float32', 2),
      outflows: view('float32', 2),
      normalization: null as never,
      output: {flows: view('float32', 4)}
    });
    expect(model.props.normalization).toBe('none');
  } finally {
    device.destroy();
  }
});
