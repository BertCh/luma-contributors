// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './visibility-graph-recipe.example';

const scenario: NetworkElementScenario = {
  id: 'gpu-network.visibility-graph-recipe',
  weights: 'lengthMeters',
  withoutRoads: true,
  roadContext: false,
  coordinateModel: 'projected',
  seed: 0,
  datasetVersion: 'synthetic-three-point-visibility-v1',
  retain: [
    'offsets',
    'neighbors',
    'weights',
    'degree',
    'grid-visits',
    'classifications',
    'pivots',
    'betweenness',
    'closeness',
    'reached-counts',
    'count',
    'requiredCount',
    'overflow',
    'candidateOverflow',
    'invalidCount',
    'approximation',
    'converged',
    'iterationLimitReached',
    'zeroWeightCycle'
  ].map(name => `visibility-graph-csr-${name}`),
  formulas: [],
  keySpaces: [
    {id: 'node', label: 'Observer', size: 3, indexSpace: '1d'},
    {id: 'node-offset', label: 'CSR boundary', size: 4, indexSpace: '1d'},
    {id: 'edge', label: 'Directed visibility slot', size: 4, indexSpace: '1d'},
    {id: 'pivot', label: 'Centrality pivot', size: 3, indexSpace: '1d'},
    {id: 'obstacle', label: 'Wall segment', size: 1, indexSpace: '1d'},
    {id: 'status', label: 'Recipe status', size: 1, indexSpace: '1d'}
  ],
  entityKeySpace: 'node',
  relations: [
    {
      id: 'node-out-edges',
      from: 'node',
      to: 'node',
      cardinality: '1:N',
      kind: 'buffer',
      timeDependent: false,
      ref: {
        layout: 'csr',
        resource: 'visibility-graph-csr-neighbors',
        offsetsResource: 'visibility-graph-csr-offsets',
        capture: 'neighbors'
      }
    }
  ],
  choices: [
    {stage: 'representation', name: 'mode', value: 'csr', sweepable: false},
    {stage: 'search', name: 'pivotCount', value: 3, sweepable: false}
  ],
  async build(context) {
    const positions = context.upload(
      'positions',
      new Float32Array([0, 0, 1, 0, 2, 0]),
      'float32x2'
    );
    const segmentWords = context.upload('segments', new Float32Array([0, 10, 2, 10]), 'float32');
    const segments = context.graph.createDataView(segmentWords.buffer, {
      format: 'float32x4',
      length: 1
    });
    const visibility = await runExample(context.graph, {
      mode: 'csr',
      positions,
      segments,
      maximumDistance: 1.1,
      edgeCapacity: 4,
      pivotCount: 3,
      laneCount: 3,
      maximumIterations: 8,
      localIterations: 4
    });
    const {status, ...outputs} = visibility.outputs;
    return {views: {positions, segments, ...outputs, ...status, pivotNodes: visibility.pivotNodes}};
  },
  capture(final) {
    if (
      final('converged').values[0] !== 1 ||
      [
        'overflow',
        'candidateOverflow',
        'invalidCount',
        'iterationLimitReached',
        'zeroWeightCycle',
        'approximation'
      ].some(name => final(name).values[0] !== 0)
    ) {
      throw new Error('Three-point visibility graph did not produce complete exact centrality');
    }
    return [
      {
        ...final('positions'),
        keySpace: 'node',
        role: 'input',
        geometry: {kind: 'points', coordinateModel: 'projected'}
      },
      {...final('segments'), keySpace: 'obstacle', role: 'input'},
      ...['degree', 'betweenness', 'gridVisits', 'viewpointClassifications'].map(name => ({
        ...final(name),
        keySpace: 'node',
        role: 'output' as const
      })),
      ...['pivotNodes', 'closeness', 'reachedCounts'].map(name => ({
        ...final(name),
        keySpace: 'pivot',
        role: 'output' as const
      })),
      {...final('offsets'), keySpace: 'node-offset', role: 'output'},
      ...['neighbors', 'weights'].map(name => ({
        ...final(name),
        keySpace: 'edge',
        role: 'output' as const
      })),
      {
        ...final('positions'),
        id: 'visibility-centrality',
        name: 'visibility-centrality',
        keySpace: 'node',
        role: 'output',
        meaning: 'Visibility path coloured by unnormalised betweenness',
        geometry: {
          kind: 'graph',
          coordinateModel: 'projected',
          offsetsCapture: 'offsets',
          targetsCapture: 'neighbors',
          layoutCapture: 'positions',
          valuesCapture: 'betweenness'
        }
      },
      ...[
        'count',
        'requiredCount',
        'overflow',
        'candidateOverflow',
        'invalidCount',
        'approximation',
        'converged',
        'iterationLimitReached',
        'zeroWeightCycle'
      ].map(name => ({
        ...final(name),
        keySpace: 'status',
        role: 'output' as const
      }))
    ];
  }
};

/** Execute the network visibility recipe and capture its adjacency, centrality and status. */
export function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Describe the same recipe graph without GPU execution. */
export function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
