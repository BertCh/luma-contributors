// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type Device, type Buffer} from '@luma.gl/core';
import {
  createGPUComputationCaptures,
  describeGPUCommandGraph,
  type GPUComputationCapture,
  type GPUComputationRecordWithSidecars
} from '@luma-contributors/analysis/computation-record';
import type {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {NETWORK_FORMULA_REFS} from '../../../gpu-network/computation-pack/network-formulas';
import {runExample} from './network-reachability.example';
import {
  assembleNetworkLensRecord,
  captureInputGeometry,
  createNetworkLensGraph,
  createNetworkRelations,
  findNetworkResource,
  getNetworkFinalCapture,
  loadChicagoLoopRoads,
  NETWORK_KEY_SPACES,
  uploadNetworkTopology,
  uploadNetworkView
} from './network-lens-support';
import {assertValidLensSidecars} from '../../validate-lens-record';
import {
  addLensViewSnapshot,
  LENS_RECORD_SOURCE_TEXT,
  readLensRunMetrics,
  readLensViewSnapshot,
  submitTimedLensGraph,
  type LensViewSnapshot
} from '../../lens-view-snapshot';

/** Fixed round budget: the single-hop proof must converge or fail visibly. */
export const NETWORK_REACHABILITY_MAXIMUM_ITERATIONS = 256;

const contributorId = 'reachability';
const retained = [
  'costs',
  'bands',
  'predecessors',
  'bandCounts',
  'converged',
  'iterationCount',
  `${contributorId}-queued-round`,
  `${contributorId}-control`
];

async function buildScenario(device: Device, chained: boolean, structureOnly: boolean) {
  const scenarioId = `gpu-network.network-reachability${chained ? '.chained' : ''}`;
  const graph = createNetworkLensGraph(device, scenarioId, retained);
  const data = loadChicagoLoopRoads();
  const ownedBuffers: Buffer[] = [];
  let compiled: CompiledGPUCommandGraph | undefined;
  const taps: {snapshot?: LensViewSnapshot} = {};
  try {
    const topology = uploadNetworkTopology(device, graph, data, {weights: 'travelTimeSeconds'});
    ownedBuffers.push(...topology.ownedBuffers);
    const startNodes = uploadNetworkView(
      device,
      graph,
      'startNodes',
      Uint32Array.of(data.startNode),
      'uint32'
    );
    ownedBuffers.push(startNodes.buffer);
    const activeIterations = uploadNetworkView(
      device,
      graph,
      'activeIterations',
      Uint32Array.of(NETWORK_REACHABILITY_MAXIMUM_ITERATIONS),
      'uint32'
    );
    ownedBuffers.push(activeIterations.buffer);
    const costBreaks = uploadNetworkView(
      device,
      graph,
      'costBreaks',
      Float32Array.of(60, 120, 180, 240, 300),
      'float32'
    );
    ownedBuffers.push(costBreaks.buffer);
    const example = await runExample(device, {
      graph,
      roads: topology,
      startNodes: startNodes.view,
      activeIterations: activeIterations.view,
      costBreaks: costBreaks.view,
      ...(chained ? {} : {localIterations: 1}),
      structureOnly,
      beforeCompile: ({output, commandNodes}) => {
        taps.snapshot = addLensViewSnapshot(graph, device, {
          id: scenarioId,
          views: {
            costs: output.costs,
            bands: output.bands,
            predecessors: output.predecessors,
            bandCounts: output.bandCounts,
            converged: output.status.converged,
            iterationCount: output.status.iterationCount,
            queuedRound: findNetworkResource(commandNodes, `${contributorId}-queued-round`),
            control: findNetworkResource(commandNodes, `${contributorId}-control`),
            offsets: topology.offsets,
            neighbors: topology.neighbors,
            nodePositions: topology.nodePositions
          }
        });
      }
    });
    compiled = example.compiled;
    const description = describeGPUCommandGraph(compiled, {
      domain: 'gpu-network',
      graph,
      sourceText: LENS_RECORD_SOURCE_TEXT
    });
    const localIterations = example.reachability.localIterations;
    const record = assembleNetworkLensRecord(device, {
      id: scenarioId,
      seed: data.startNode,
      variant: chained ? 'chained' : 'single-hop',
      graphs: [
        {
          ...description,
          graph: {
            ...description.graph,
            attrs: {
              ...description.graph.attrs,
              localIterations,
              maximumIterations: NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
              observation: chained
                ? 'end capture only'
                : 'end capture only; per-round frames need taps authored between rounds'
            }
          }
        }
      ],
      formulas: Object.values(NETWORK_FORMULA_REFS),
      choices: [
        {
          stage: 'search',
          name: 'localIterations',
          value: localIterations,
          alternatives: [chained ? 1 : 32],
          sweepable: true
        }
      ],
      axes: {
        axes: [
          {
            id: 'entity',
            label: 'network node',
            kind: 'data',
            keySpace: 'node',
            size: data.nodeCount
          },
          {
            id: 'iteration',
            label: 'completed relax rounds',
            kind: 'execution',
            size: NETWORK_REACHABILITY_MAXIMUM_ITERATIONS + 1
          }
        ],
        keySpaces: NETWORK_KEY_SPACES,
        relations: createNetworkRelations(topology, `${contributorId}-queues`)
      }
    });
    // The example invokes beforeCompile exactly once, before it compiles.
    assert(taps.snapshot);
    return {example, compiled, snapshot: taps.snapshot, record, data, ownedBuffers};
  } catch (error) {
    taps.snapshot?.arena.destroy();
    compiled?.destroy();
    for (const buffer of ownedBuffers) buffer.destroy();
    throw error;
  }
}

/** Converts exact queuedRound readbacks into named round stamps; rejects tie-phase contamination. */
export function makeReachabilityStamp(
  queued: GPUComputationCapture,
  maximumIterations: number,
  chained: boolean
): GPUComputationCapture {
  const values = Float32Array.from(queued.values, stamp => {
    // A later tie phase shares this buffer. Positive road weights must leave it untouched.
    assert(stamp <= maximumIterations + 1);
    return stamp >= 1 ? stamp - 1 : NaN;
  });
  return {
    ...queued,
    id: chained ? 'lastQueuedRound' : 'lastImprovedRound',
    name: chained ? 'lastQueuedRound' : 'lastImprovedRound',
    keySpace: 'node',
    elementType: 'f32',
    values,
    meaning: chained
      ? 'last global-queue round (chained hops inside a round are not stamped)'
      : 'last-improved round (frontier queuedRound stamp − 1; exact because localIterations = 1)'
  };
}

/** Runs either proof variant; all values are real GPU readbacks, as end-of-run GPU readbacks. */
export async function runReachabilityScenario(
  device: Device,
  chained: boolean,
  options?: {signal?: AbortSignal}
): Promise<GPUComputationRecordWithSidecars> {
  options?.signal?.throwIfAborted();
  const built = await buildScenario(device, chained, false);
  const {compiled, record, data} = built;
  try {
    const submission = submitTimedLensGraph(device, compiled, {
      parameters: undefined,
      coalesceComputePasses: false
    });
    const timedMetrics = await readLensRunMetrics(submission, {
      runId: record.runId,
      graphId: record.graphs[0].id,
      taps: 1,
      retained
    });
    const captured = await readLensViewSnapshot(built.snapshot, {graphId: record.graphs[0].id});
    options?.signal?.throwIfAborted();
    const final = (name: string) => getNetworkFinalCapture(captured, name);
    if (final('converged').values[0] === 0)
      throw new Error('Network reachability did not converge within maximumIterations = 256');
    const iterationCount = final('iterationCount').values[0];
    const stamp = makeReachabilityStamp(
      final('queuedRound'),
      NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
      chained
    );
    const snapshots: GPUComputationCapture[] = [];
    snapshots.push(
      ...[
        'costs',
        'bands',
        'predecessors',
        'bandCounts',
        'converged',
        'iterationCount',
        'queuedRound',
        'control'
      ].map(name => ({
        ...final(name),
        ...(['costs', 'bands', 'predecessors', 'queuedRound'].includes(name)
          ? {keySpace: 'node'}
          : {})
      }))
    );
    snapshots.push(...captureInputGeometry(captured, [data.startNode]));
    const positions = final('nodePositions');
    snapshots.push({
      ...positions,
      id: 'reached-nodes',
      name: 'reached-nodes',
      keySpace: 'node',
      role: 'output',
      geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: 'costs'},
      meaning:
        'output: all nodes coloured by final travel time; +Infinity is no data, including beyond the displayed bands'
    });
    const captures = createGPUComputationCaptures({
      runId: record.runId,
      snapshots,
      stamps: [stamp]
    });
    const metrics = [
      {
        ...timedMetrics,
        series: [
          {
            name: 'frontier-size',
            axis: 'iteration',
            unit: 'nodes',
            values: Array.from(final('control').values).slice(4, 5 + iterationCount)
          }
        ]
      }
    ];
    const result = {...record, captures, metrics};
    assertValidLensSidecars(record, result);
    return result;
  } finally {
    built.snapshot.arena.destroy();
    compiled.destroy();
    for (const buffer of built.ownedBuffers) buffer.destroy();
  }
}

/** Describes either variant on a null device, without submitting or reading buffers. */
export async function describeReachabilityStructure(device: Device, chained: boolean) {
  const built = await buildScenario(device, chained, true);
  try {
    return built.record;
  } finally {
    built.snapshot.arena.destroy();
    built.compiled.destroy();
    for (const buffer of built.ownedBuffers) buffer.destroy();
  }
}

/** Captures the single-hop wavefront proof as end-of-run readbacks. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  return runReachabilityScenario(device, false, options);
}

/** Builds the single-hop structure without execution. */
export async function describeStructure(device: Device) {
  return describeReachabilityStructure(device, false);
}
