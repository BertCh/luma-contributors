// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkCostMatrix,
  GPUNetworkServiceAreas,
  GPUNetworkCostWeights,
  GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH,
  getGPUNetworkCostWeightsParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {addKernelPass} from '../../engine/mode-kernels';
import type {SceneContext, SceneInstance} from '../scene';
import {findStation, loadRailGraph} from './transit-data';

export type RailSupportPlanningOptions = {
  direction: 'inbound' | 'outbound';
  costLimitMinutes: number;
  decay: 'none' | 'exponential' | 'power' | 'gaussian';
  beta: number;
  display: 'assignments' | 'weightedReach';
};

const HUB_NAMES = [
  'Amsterdam Centraal',
  'Utrecht Centraal',
  'Rotterdam Centraal',
  'Zwolle'
] as const;
const HUB_COLORS = [
  [36, 132, 173, 255],
  [238, 132, 43, 255],
  [135, 91, 177, 255],
  [45, 153, 112, 255]
] as const;

function makeReverseGraph(
  offsets: Uint32Array,
  targets: Uint32Array,
  weights: Float32Array,
  nodeCount: number
) {
  const reverseOffsets = new Uint32Array(nodeCount + 1);
  for (const target of targets) reverseOffsets[target + 1]++;
  for (let node = 0; node < nodeCount; node++) reverseOffsets[node + 1] += reverseOffsets[node];
  const cursors = reverseOffsets.slice(0, nodeCount);
  const reverseTargets = new Uint32Array(targets.length),
    reverseWeights = new Float32Array(weights.length);
  for (let source = 0; source < nodeCount; source++)
    for (let edge = offsets[source]; edge < offsets[source + 1]; edge++) {
      const slot = cursors[targets[edge]]++;
      reverseTargets[slot] = source;
      reverseWeights[slot] = weights[edge];
    }
  return {offsets: reverseOffsets, targets: reverseTargets, weights: reverseWeights};
}

export async function createRailSupportPlanning(
  ctx: SceneContext<RailSupportPlanningOptions>
): Promise<SceneInstance<RailSupportPlanningOptions>> {
  const rail = loadRailGraph(ctx.datasets.get('gtfs-nl-rail-graph'));
  const hubNodes = HUB_NAMES.map(name => findStation(rail, name));
  const {device} = ctx;
  const {nodeCount, edgeCount} = rail;
  const resources = new SpatialAnalysisResources(device, 'rail-support');
  const facilities = resources.createBuffer('candidate-hubs', Uint32Array.from(hubNodes));
  const hubPositions = resources.createBuffer(
    'candidate-hub-positions',
    Float32Array.from(
      hubNodes.flatMap(node => [...rail.nodePositions.subarray(node * 2, node * 2 + 2)])
    ),
    Buffer.VERTEX
  );
  const stationBuffer = resources.createBuffer(
    'station-positions',
    rail.nodePositions,
    Buffer.VERTEX
  );
  const railSegmentsBuffer = resources.createBuffer(
    'rail-network-segments',
    rail.segments,
    Buffer.VERTEX
  );
  const assignmentBuffer = resources.createBuffer('assignments', nodeCount * 4);
  const serviceCostBuffer = resources.createBuffer('service-costs', nodeCount * 4);
  const serviceStatusBuffer = resources.createBuffer('service-status', 4);
  const countsBuffer = resources.createBuffer('facility-counts', HUB_NAMES.length * 4);
  const matrixBuffer = resources.createBuffer('hub-cost-matrix', HUB_NAMES.length * nodeCount * 4);
  const minuteMatrixBuffer = resources.createBuffer(
    'hub-cost-matrix-minutes',
    HUB_NAMES.length * nodeCount * 4
  );
  const matrixStatusBuffer = resources.createBuffer('matrix-status', 4);
  const weightOffsetsBuffer = resources.createBuffer('weight-offsets', (nodeCount + 1) * 4);
  const weightNeighborsBuffer = resources.createBuffer(
    'weight-neighbors',
    HUB_NAMES.length * nodeCount * 4
  );
  const weightValuesBuffer = resources.createBuffer(
    'weight-values',
    HUB_NAMES.length * nodeCount * 4
  );
  const weightedReachBuffer = resources.createBuffer('weighted-reach', nodeCount * 4);
  const weightStatusBuffer = resources.createBuffer('weight-status', 8);
  const serviceLimitBuffer = resources.createBuffer(
    'service-cost-limit',
    Float32Array.of(ctx.options.costLimitMinutes * 60)
  );
  const weightParametersBuffer = resources.createBuffer(
    'weight-parameters',
    getGPUNetworkCostWeightsParameterValues({
      costLimit: ctx.options.costLimitMinutes,
      decay: ctx.options.decay,
      beta: ctx.options.beta,
      minimumCost: 1
    })
  );
  const forward = {
    offsets: resources.createBuffer('forward-offsets', rail.offsets),
    neighbors: resources.createBuffer('forward-neighbors', rail.targets),
    weights: resources.createBuffer('forward-travel-times', rail.travelTime)
  };
  const reverseGraph = makeReverseGraph(rail.offsets, rail.targets, rail.travelTime, nodeCount);
  const reverse = {
    offsets: resources.createBuffer('reverse-offsets', reverseGraph.offsets),
    neighbors: resources.createBuffer('reverse-neighbors', reverseGraph.targets),
    weights: resources.createBuffer('reverse-travel-times', reverseGraph.weights)
  };

  function buildGraph(directionValue: RailSupportPlanningOptions['direction']) {
    const graph = new GPUCommandGraph<void>(device, {id: 'rail-support-planning'});
    const view = createGraphBufferImporter(graph);
    const topology = directionValue === 'inbound' ? reverse : forward;
    const matrixTopology = {
      offsets: view(topology.offsets, 'uint32', nodeCount + 1),
      neighbors: view(topology.neighbors, 'uint32', edgeCount),
      weights: view(topology.weights, 'float32', edgeCount)
    };
    graph.add(
      new GPUNetworkCostMatrix({
        id: 'candidate-desk-costs',
        topology: matrixTopology,
        startNodes: view(facilities, 'uint32', HUB_NAMES.length),
        laneCount: HUB_NAMES.length,
        maximumIterations: 16,
        localIterations: 32,
        output: {
          costs: view(matrixBuffer, 'float32', HUB_NAMES.length * nodeCount),
          status: {converged: view(matrixStatusBuffer, 'uint32', 1)}
        }
      })
    );
    addKernelPass(graph, {
      id: 'convert-desk-costs-to-minutes',
      invocationCount: HUB_NAMES.length * nodeCount,
      bindings: [
        {
          name: 'seconds',
          view: view(matrixBuffer, 'float32', HUB_NAMES.length * nodeCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'minutes',
          view: view(minuteMatrixBuffer, 'float32', HUB_NAMES.length * nodeCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      // Keep exact whole-minute boundaries exact when the backend lowers division to reciprocal multiplication.
      body: `let value = seconds[secondsOffset + index];
        minutes[minutesOffset + index] = value;
        if (value >= 0.0 && value < 4294967040.0) {
          let wholeMinutes = u32(value) / 60u;
          let remainderSeconds = value - f32(wholeMinutes * 60u);
          minutes[minutesOffset + index] = f32(wholeMinutes) + remainderSeconds / 60.0;
        }`
    });
    graph.add(
      new GPUNetworkServiceAreas({
        id: 'nearest-desk-service-areas',
        topology: matrixTopology,
        facilities: view(facilities, 'uint32', HUB_NAMES.length),
        costLimit: view(serviceLimitBuffer, 'float32', 1),
        maximumIterations: 16,
        maximumLabelIterations: 16,
        localIterations: 32,
        output: {
          assignments: view(assignmentBuffer, 'uint32', nodeCount),
          costs: view(serviceCostBuffer, 'float32', nodeCount),
          facilityNodeCounts: view(countsBuffer, 'uint32', HUB_NAMES.length),
          status: {converged: view(serviceStatusBuffer, 'uint32', 1)}
        }
      })
    );
    graph.add(
      new GPUNetworkCostWeights({
        id: 'candidate-desk-decay-weights',
        costs: view(minuteMatrixBuffer, 'float32', HUB_NAMES.length * nodeCount),
        rowCount: HUB_NAMES.length,
        orientation: 'columns',
        parameters: view(
          weightParametersBuffer,
          'float32',
          GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH
        ),
        output: {
          offsets: view(weightOffsetsBuffer, 'uint32', nodeCount + 1),
          neighbors: view(weightNeighborsBuffer, 'uint32', HUB_NAMES.length * nodeCount),
          weights: view(weightValuesBuffer, 'float32', HUB_NAMES.length * nodeCount),
          status: {overflow: view(weightStatusBuffer, 'uint32', 1)}
        }
      })
    );
    addKernelPass(graph, {
      id: 'sum-desk-weights-by-station',
      invocationCount: nodeCount,
      bindings: [
        {
          name: 'offsets',
          view: view(weightOffsetsBuffer, 'uint32', nodeCount + 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'weights',
          view: view(weightValuesBuffer, 'float32', HUB_NAMES.length * nodeCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'reach',
          view: view(weightedReachBuffer, 'float32', nodeCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `var total = 0.0; for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) { total += weights[weightsOffset + slot]; } reach[reachOffset + index] = total;`
    });
    return graph;
  }
  const inboundGraph = buildGraph('inbound');
  const outboundGraph = buildGraph('outbound');
  const inboundCompiled = resources.track(await inboundGraph.compileAsync());
  const outboundCompiled = resources.track(await outboundGraph.compileAsync());
  let compiled: CompiledGPUCommandGraph<void> =
    ctx.options.direction === 'inbound' ? inboundCompiled : outboundCompiled;
  let direction = ctx.options.direction;
  let dirty = true,
    generation = 1,
    pendingGeneration = 0,
    destroyed = false;
  let outputReady = false;
  let lastAssignments: Uint32Array | null = null;
  let lastCosts: Float32Array | null = null;
  let lastWeightedReach: Float32Array | null = null;
  const reader = new SummaryReader(
    resources,
    'rail-support-summary',
    [
      {buffer: assignmentBuffer, size: nodeCount * 4},
      {buffer: serviceCostBuffer, size: nodeCount * 4},
      {buffer: serviceStatusBuffer, size: 4},
      {buffer: countsBuffer, size: HUB_NAMES.length * 4},
      {buffer: matrixStatusBuffer, size: 4},
      {buffer: weightOffsetsBuffer, size: (nodeCount + 1) * 4},
      {buffer: weightStatusBuffer, size: 4},
      {buffer: weightedReachBuffer, size: nodeCount * 4}
    ],
    bytes => {
      if (destroyed || pendingGeneration !== generation) return;
      const words = new Uint32Array(bytes);
      const assignments = words.slice(0, nodeCount);
      const floats = new Float32Array(bytes);
      outputReady = true;
      lastAssignments = assignments;
      lastCosts = floats.slice(nodeCount, nodeCount * 2);
      const serviceStatus = words[nodeCount * 2];
      const counts = words.slice(nodeCount * 2 + 1, nodeCount * 2 + 1 + HUB_NAMES.length);
      const matrixStatus = words[nodeCount * 2 + 1 + HUB_NAMES.length];
      const weightStart = nodeCount * 2 + 2 + HUB_NAMES.length;
      const offsets = words.slice(weightStart, weightStart + nodeCount + 1);
      const weightOverflow = words[weightStart + nodeCount + 1];
      const weightedReachStart = weightStart + nodeCount + 2;
      const assignedCount = counts.reduce((sum, count) => sum + count, 0);
      const converged = serviceStatus === 1 && matrixStatus === 1 && !weightOverflow;
      outputReady = converged;
      lastWeightedReach = converged
        ? floats.slice(weightedReachStart, weightedReachStart + nodeCount)
        : null;
      ctx.setReadout(
        'stationsServed',
        converged
          ? `${assignedCount} / ${nodeCount} stations`
          : 'withheld until GPU searches converge'
      );
      ctx.setReadout(
        'unassigned',
        converged ? `${nodeCount - assignedCount} stations` : 'withheld'
      );
      ctx.setReadout(
        'weightedPairs',
        converged
          ? `${offsets[nodeCount]} station-desk pairs · total weight ${(lastWeightedReach?.reduce((sum, value) => sum + value, 0) ?? 0).toFixed(1)}`
          : 'withheld'
      );
      ctx.setReadout(
        'deskSummary',
        converged
          ? Array.from(
              counts,
              (count, index) => `${HUB_NAMES[index].replace(' Centraal', '')}: ${count}`
            ).join(' · ')
          : 'withheld until GPU searches converge'
      );
      ctx.setReadout(
        'travelModel',
        ctx.options.direction === 'inbound'
          ? 'Station to desk · reverse CSR · median scheduled time'
          : 'Desk to station · forward CSR · median scheduled time'
      );
      ctx.setReadout(
        'solver',
        converged
          ? 'Service areas and cost weights converged'
          : 'GPU output incomplete; counts withheld'
      );
      ctx.requestLayers();
    }
  );
  function updateParameters() {
    serviceLimitBuffer.write(Float32Array.of(ctx.options.costLimitMinutes * 60));
    weightParametersBuffer.write(
      getGPUNetworkCostWeightsParameterValues({
        costLimit: ctx.options.costLimitMinutes,
        decay: ctx.options.decay,
        beta: ctx.options.beta,
        minimumCost: 1
      })
    );
    generation++;
    dirty = true;
    outputReady = false;
    lastAssignments = null;
    lastCosts = null;
    lastWeightedReach = null;
    reader.markStale();
    ctx.setReadout('stationsServed', 'updating GPU service areas');
    ctx.setReadout('unassigned', '—');
    ctx.setReadout('weightedPairs', '—');
    ctx.setReadout('deskSummary', '—');
    ctx.setReadout('solver', 'Waiting for GPU graph analysis');
  }
  ctx.setReadout('solver', 'Waiting for GPU graph analysis');
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      if (!reader.isPending && generation !== pendingGeneration) pendingGeneration = generation;
      reader.flush(commandEncoder);
    },
    setOption(id) {
      if (id === 'direction' && direction !== ctx.options.direction) {
        direction = ctx.options.direction;
        compiled = direction === 'inbound' ? inboundCompiled : outboundCompiled;
        generation++;
        dirty = true;
        outputReady = false;
        lastAssignments = null;
        lastCosts = null;
        lastWeightedReach = null;
        reader.markStale();
        ctx.setReadout('stationsServed', 'updating GPU service areas');
        ctx.setReadout('unassigned', '—');
        ctx.setReadout('weightedPairs', '—');
        ctx.setReadout('deskSummary', '—');
        ctx.setReadout('solver', 'Waiting for GPU graph analysis');
      } else if (id === 'costLimitMinutes' || id === 'decay' || id === 'beta') updateParameters();
      ctx.requestLayers();
    },
    getLayers() {
      const palette: [number, number, number, number][] = HUB_COLORS.map(color => [...color]);
      const assignmentLayer = outputReady
        ? new SpatialAnalysisPointLayer({
            id: 'rail-support-station-assignments',
            analysisFrame: rail.frame,
            positions: stationBuffer,
            instanceCount: nodeCount,
            values: assignmentBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette,
            noDataColor: [150, 154, 158, 180],
            radiusPixels: 4,
            outlineColor: [255, 255, 255, 160],
            outlineWidthPixels: 0.4
          })
        : new SpatialAnalysisPointLayer({
            id: 'rail-support-stations-pending',
            analysisFrame: rail.frame,
            positions: stationBuffer,
            instanceCount: nodeCount,
            radiusPixels: 3,
            color: [131, 140, 148, 180]
          });
      const weightedLayer = new SpatialAnalysisPointLayer({
        id: 'rail-support-weighted-reach',
        analysisFrame: rail.frame,
        positions: stationBuffer,
        instanceCount: nodeCount,
        values: weightedReachBuffer,
        valueFormat: 'float32',
        colormap: 'cividis',
        valueRange: [0, HUB_NAMES.length],
        noDataColor: [150, 154, 158, 180],
        radiusPixels: 4,
        outlineColor: [255, 255, 255, 160],
        outlineWidthPixels: 0.4
      });
      return [
        new SpatialAnalysisSegmentLayer({
          id: 'rail-support-network-context',
          analysisFrame: rail.frame,
          segments: railSegmentsBuffer,
          instanceCount: edgeCount,
          widthPixels: 0.7,
          color: [112, 129, 145, 48]
        }),
        outputReady && ctx.options.display === 'weightedReach' ? weightedLayer : assignmentLayer,
        new SpatialAnalysisPointLayer({
          id: 'rail-support-candidate-desks',
          analysisFrame: rail.frame,
          positions: hubPositions,
          instanceCount: HUB_NAMES.length,
          radiusPixels: 8,
          color: [255, 250, 235, 255],
          outlineColor: [28, 35, 42, 255],
          outlineWidthPixels: 2,
          shape: 'star'
        })
      ] as Layer[];
    },
    getTooltip(event) {
      if (!outputReady) return null;
      const viewport = ctx.getViewport();
      if (!viewport) return null;
      let closest = 12 * 12,
        nearest = -1;
      for (let node = 0; node < nodeCount; node++) {
        const pixel = viewport.project([rail.nodeLngLat[node * 2], rail.nodeLngLat[node * 2 + 1]]);
        const distance = (pixel[0] - event.pixel[0]) ** 2 + (pixel[1] - event.pixel[1]) ** 2;
        if (distance < closest) {
          closest = distance;
          nearest = node;
        }
      }
      if (nearest < 0) return null;
      const assigned = lastAssignments?.[nearest] ?? 0xffffffff;
      const cost = lastCosts?.[nearest];
      const weightedValue = lastWeightedReach?.[nearest];
      return {
        title: rail.names[nearest],
        subtitle: 'Hypothetical rail support assignment',
        rows: [
          {
            label: 'Assigned desk',
            value:
              assigned < HUB_NAMES.length
                ? HUB_NAMES[assigned].replace(' Centraal', '')
                : 'none within limit'
          },
          {
            label: 'Scheduled rail cost',
            value:
              cost !== undefined && Number.isFinite(cost)
                ? `${(cost / 60).toFixed(1)} min`
                : 'unreachable'
          },
          {
            label: 'Summed desk weights',
            value: weightedValue === undefined ? 'waiting for GPU' : weightedValue.toFixed(2)
          },
          {
            label: 'Travel direction',
            value: ctx.options.direction === 'inbound' ? 'station to desk' : 'desk to station'
          },
          {label: 'Travel-time limit', value: `${ctx.options.costLimitMinutes} min`}
        ],
        note: 'One unit per station is an illustrative demand assumption. The graph uses median scheduled in-vehicle link times and perfect transfers.'
      };
    },
    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
