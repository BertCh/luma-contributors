// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {
  addNetworkDisruptionRecipe,
  GPUNetworkStructure,
  GPUNetworkSubgraphFilter,
  GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
  getGPUNetworkAccessibilityParameterValues
} from '@luma-contributors/graph/gpu-network';
import {
  GPUCommandGraph,
  GPUElementwise,
  type CompiledGPUCommandGraph
} from '@luma-contributors/graph/gpu-core';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {findStation, loadRailGraph} from './transit-data';

export type RailResilienceOptions = {
  origin: string;
  corridor: string;
  event: 'closure' | 'slowdown';
  slowdownMinutes: number;
  budgetMinutes: number;
  trainTypes: 'all' | 'fast' | 'stopping';
  dwellSeconds: number;
  display: 'baseline' | 'scenario' | 'delta' | 'unreachable';
  showNetwork: boolean;
  showStations: boolean;
};

type Corridor = {label: string; a: string; b: string};
const CORRIDORS: readonly Corridor[] = [
  {label: 'Zwolle interchange offline', a: 'Zwolle', b: ''},
  {label: 'Utrecht interchange offline', a: 'Utrecht Centraal', b: ''},
  {label: 'Bodegraven–Alphen a/d Rijn', a: 'Bodegraven', b: 'Alphen a/d Rijn'},
  {label: 'Utrecht–Amsterdam', a: 'Utrecht Centraal', b: 'Amsterdam Centraal'},
  {label: 'Rotterdam–Delft', a: 'Rotterdam Centraal', b: 'Delft'},
  {label: 'Gouda–Utrecht', a: 'Gouda', b: 'Utrecht Centraal'}
];
const CHANGE_COLOR = [232, 112, 50, 255] as const;
const QUIET_INK = [45, 58, 70, 95] as const;

function corridorEdges(rail: ReturnType<typeof loadRailGraph>, corridor: Corridor): number[] {
  const first = findStation(rail, corridor.a);
  const second = corridor.b ? findStation(rail, corridor.b) : -1;
  const edges: number[] = [];
  for (let edge = 0; edge < rail.edgeCount; edge++) {
    if (
      second < 0
        ? rail.sources[edge] === first || rail.targets[edge] === first
        : (rail.sources[edge] === first && rail.targets[edge] === second) ||
          (rail.sources[edge] === second && rail.targets[edge] === first)
    )
      edges.push(edge);
  }
  return edges;
}

/** Compare one station's rail access under paired baseline and disrupted GPU searches. */
export async function createRailResilience(
  ctx: SceneContext<RailResilienceOptions>
): Promise<SceneInstance<RailResilienceOptions>> {
  const rail = loadRailGraph(ctx.datasets.get('gtfs-nl-rail-graph'));
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'rail-resilience');
  const nodeCount = rail.nodeCount;
  const edgeCount = rail.edgeCount;
  const targets = Uint32Array.from({length: nodeCount}, (_, index) => index);
  const targetBuffer = resources.createBuffer('target-stations', targets);
  const startNodesBuffer = resources.createBuffer('start-station', Uint32Array.of(0));
  const stationBuffer = resources.createBuffer('stations', rail.nodePositions, Buffer.VERTEX);
  const segmentBuffer = resources.createBuffer('rail-links', rail.segments, Buffer.VERTEX);
  const originPositionBuffer = resources.createBuffer('origin-position', 8, Buffer.VERTEX);
  const stressedSegmentsBuffer = resources.createBuffer(
    'stressed-links',
    edgeCount * 16,
    Buffer.VERTEX
  );
  const weightsBuffer = resources.createBuffer('baseline-weights', rail.travelTime);
  const edgeMaskBuffer = resources.createBuffer('edge-mask', new Uint32Array(edgeCount).fill(1));
  const replacementWeightsBuffer = resources.createBuffer('replacement-weights', rail.travelTime);
  const opportunitiesBuffer = resources.createBuffer(
    'station-opportunities',
    new Float32Array(nodeCount).fill(1)
  );
  const parametersBuffer = resources.createParameterBuffer(
    'accessibility-parameters',
    'float32',
    GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
    getGPUNetworkAccessibilityParameterValues({threshold: ctx.options.budgetMinutes * 60})
  );

  const costsLength = nodeCount;
  const baselineCostsBuffer = resources.createBuffer('baseline-costs', costsLength * 4);
  const scenarioCostsBuffer = resources.createBuffer('scenario-costs', costsLength * 4);
  const costDifferencesBuffer = resources.createBuffer('cost-differences', costsLength * 4);
  const newlyUnreachableBuffer = resources.createBuffer('newly-unreachable', costsLength * 4);
  const baselineAccessibilityBuffer = resources.createBuffer('baseline-accessibility', 4);
  const scenarioAccessibilityBuffer = resources.createBuffer('scenario-accessibility', 4);
  const accessibilityDifferencesBuffer = resources.createBuffer('accessibility-differences', 4);
  const topologyOffsetsBuffer = resources.createBuffer('topology-offsets', rail.offsets);
  const topologyNeighborsBuffer = resources.createBuffer('topology-neighbors', rail.targets);
  const liveNodeMaskBuffer = resources.createBuffer('live-node-mask', new Uint32Array(nodeCount));
  const liveEdgeMaskBuffer = resources.createBuffer('live-edge-mask', new Uint32Array(edgeCount));
  const inducedOffsetsBuffer = resources.createBuffer('induced-offsets', (nodeCount + 1) * 4);
  const inducedNeighborsBuffer = resources.createBuffer('induced-neighbors', edgeCount * 4);
  const inducedSourceSlotsBuffer = resources.createBuffer('induced-source-slots', edgeCount * 4);
  const inducedOverflowBuffer = resources.createBuffer('induced-overflow', 4);
  const componentLabelsBuffer = resources.createBuffer('components', nodeCount * 4);
  const baselineStatusBuffer = resources.createBuffer('baseline-status', 4);
  const scenarioStatusBuffer = resources.createBuffer('scenario-status', 4);

  const graph = new GPUCommandGraph<void>(device, {id: 'rail-resilience'});
  const view = createGraphBufferImporter(graph);
  const topology = {
    offsets: view(topologyOffsetsBuffer, 'uint32', nodeCount + 1),
    neighbors: view(topologyNeighborsBuffer, 'uint32', edgeCount)
  };
  graph.add(
    new GPUNetworkSubgraphFilter({
      id: 'live-rail-subgraph',
      topology,
      directed: true,
      edgeMask: view(edgeMaskBuffer, 'uint32', edgeCount),
      output: {
        nodeMask: view(liveNodeMaskBuffer, 'uint32', nodeCount),
        edgeMask: view(liveEdgeMaskBuffer, 'uint32', edgeCount),
        inducedTopology: {
          offsets: view(inducedOffsetsBuffer, 'uint32', nodeCount + 1),
          neighbors: view(inducedNeighborsBuffer, 'uint32', edgeCount),
          sourceSlots: view(inducedSourceSlotsBuffer, 'uint32', edgeCount),
          status: {overflow: view(inducedOverflowBuffer, 'uint32', 1)}
        }
      }
    })
  );
  graph.add(
    new GPUNetworkStructure({
      id: 'live-rail-components',
      topology: {
        offsets: view(inducedOffsetsBuffer, 'uint32', nodeCount + 1),
        neighbors: view(inducedNeighborsBuffer, 'uint32', edgeCount)
      },
      output: {strongComponents: view(componentLabelsBuffer, 'uint32', nodeCount)}
    })
  );
  const result = addNetworkDisruptionRecipe(graph, {
    id: 'rail-disruption',
    rowCount: 1,
    search: {
      topology: {
        offsets: topology.offsets,
        neighbors: topology.neighbors,
        weights: view(weightsBuffer, 'float32', edgeCount)
      },
      startNodes: view(startNodesBuffer, 'uint32', 1),
      targetNodes: view(targetBuffer, 'uint32', nodeCount),
      laneCount: 1,
      maximumIterations: 24,
      localIterations: 32
    },
    edgeMask: view(edgeMaskBuffer, 'uint32', edgeCount),
    replacementCosts: view(replacementWeightsBuffer, 'float32', edgeCount),
    opportunityWeights: view(opportunitiesBuffer, 'float32', nodeCount),
    parameters: parametersBuffer.importToGraph(graph),
    outputs: {
      baselineCosts: view(baselineCostsBuffer, 'float32', costsLength),
      scenarioCosts: view(scenarioCostsBuffer, 'float32', costsLength),
      costDifferences: view(costDifferencesBuffer, 'float32', costsLength),
      newlyUnreachable: view(newlyUnreachableBuffer, 'uint32', costsLength),
      baselineAccessibility: view(baselineAccessibilityBuffer, 'float32', 1),
      scenarioAccessibility: view(scenarioAccessibilityBuffer, 'float32', 1),
      accessibilityDifferences: view(accessibilityDifferencesBuffer, 'float32', 1)
    }
  });
  graph.add(
    new GPUElementwise({
      id: 'publish-baseline-convergence',
      input: result.status.stages[0].status.converged!,
      output: view(baselineStatusBuffer, 'uint32', 1),
      operation: 'copy'
    })
  );
  graph.add(
    new GPUElementwise({
      id: 'publish-scenario-convergence',
      input: result.status.stages[1].status.converged!,
      output: view(scenarioStatusBuffer, 'uint32', 1),
      operation: 'copy'
    })
  );
  const compiled: CompiledGPUCommandGraph<void> = resources.track(await graph.compileAsync());
  let originNode = findStation(rail, ctx.options.origin);
  let generation = 0;
  let pendingGeneration = 0;
  let dirty = true;
  let destroyed = false;
  let summary: {
    baseline: Float32Array;
    scenario: Float32Array;
    delta: Float32Array;
    unreachable: Uint32Array;
    baselineAccess: number;
    scenarioAccess: number;
    converged: boolean;
  } | null = null;
  function markDirty(): void {
    generation++;
    dirty = true;
    summary = null;
    ctx.setReadout('solver', 'Waiting for GPU routing');
    ctx.setReadout('baselineReach', '—');
    ctx.setReadout('scenarioReach', '—');
    ctx.setReadout('newlyUnreachable', '—');
    ctx.setReadout('reachLoss', '—');
    ctx.setReadout('medianAdded', '—');
    ctx.setReadout('affectedStations', '—');
    ctx.setReadout('components', '—');
    ctx.setChart('impact', null);
  }
  const updateCounts = () => {
    if (!summary) return;
    if (!summary.converged) {
      ctx.setReadout('solver', 'Routing did not converge; comparison withheld');
      ctx.setReadout('baselineReach', 'withheld');
      ctx.setReadout('scenarioReach', 'withheld');
      ctx.setReadout('newlyUnreachable', 'withheld');
      ctx.setReadout('reachLoss', 'withheld');
      ctx.setReadout('medianAdded', 'withheld');
      ctx.setReadout('affectedStations', 'withheld');
      return;
    }
    let newlyUnreachable = 0;
    const deltas: number[] = [];
    for (let station = 0; station < nodeCount; station++) {
      newlyUnreachable += summary.unreachable[station];
      if (
        summary.delta[station] > 0 &&
        Number.isFinite(summary.delta[station]) &&
        Number.isFinite(summary.scenario[station])
      )
        deltas.push(summary.delta[station] / 60);
    }
    deltas.sort((first, second) => first - second);
    const middle = Math.floor(deltas.length / 2);
    const median = !deltas.length
      ? Number.NaN
      : deltas.length % 2
        ? deltas[middle]
        : (deltas[middle - 1] + deltas[middle]) / 2;
    const before = Math.round(summary.baselineAccess);
    const after = Math.round(summary.scenarioAccess);
    ctx.setReadout('baselineReach', `${before} of ${nodeCount} stations`);
    ctx.setReadout('scenarioReach', `${after} of ${nodeCount} stations`);
    ctx.setReadout(
      'newlyUnreachable',
      `${newlyUnreachable} ${newlyUnreachable === 1 ? 'station' : 'stations'}`
    );
    ctx.setReadout(
      'reachLoss',
      `${Math.max(0, before - after)} ${before - after === 1 ? 'station' : 'stations'}`
    );
    ctx.setReadout(
      'affectedStations',
      `${deltas.length} ${deltas.length === 1 ? 'station' : 'stations'}`
    );
    ctx.setReadout('medianAdded', Number.isFinite(median) ? `${Math.round(median)} min` : '—');
    ctx.setReadout('solver', 'Both routing stages converged');
    ctx.setChart('impact', {
      kind: 'bars',
      height: 110,
      values: [before, after, newlyUnreachable],
      labels: ['Before', 'After', 'Newly unreachable'],
      highlight: [2],
      yLabel: 'stations',
      description:
        'Station counts within the selected time budget and count newly disconnected from the selected hub.'
    });
    ctx.requestLayers();
  };

  const reader = new SummaryReader(
    resources,
    'scenario-summary',
    [
      {buffer: baselineCostsBuffer, size: costsLength * 4},
      {buffer: scenarioCostsBuffer, size: costsLength * 4},
      {buffer: costDifferencesBuffer, size: costsLength * 4},
      {buffer: newlyUnreachableBuffer, size: costsLength * 4},
      {buffer: componentLabelsBuffer, size: nodeCount * 4},
      {buffer: inducedOverflowBuffer, size: 4},
      {buffer: baselineStatusBuffer, size: 4},
      {buffer: scenarioStatusBuffer, size: 4},
      {buffer: baselineAccessibilityBuffer, size: 4},
      {buffer: scenarioAccessibilityBuffer, size: 4},
      {buffer: accessibilityDifferencesBuffer, size: 4}
    ],
    bytes => {
      if (destroyed || pendingGeneration !== generation) return;
      const floatValues = new Float32Array(bytes);
      const wordValues = new Uint32Array(bytes);
      const componentLabels = wordValues.slice(costsLength * 4, costsLength * 5);
      const components = new Set(componentLabels).size;
      const statusOffset = nodeCount * 5 + 1;
      const converged = wordValues[statusOffset] !== 0 && wordValues[statusOffset + 1] !== 0;
      const accessibilityOffset = nodeCount * 5 + 3;
      summary = {
        baseline: floatValues.slice(0, costsLength),
        scenario: floatValues.slice(costsLength, costsLength * 2),
        delta: floatValues.slice(costsLength * 2, costsLength * 3),
        unreachable: wordValues.slice(costsLength * 3, costsLength * 4),
        baselineAccess: floatValues[accessibilityOffset],
        scenarioAccess: floatValues[accessibilityOffset + 1],
        converged
      };
      ctx.setReadout(
        'components',
        `${components} strongly connected ${components === 1 ? 'component' : 'components'} after filtering`
      );
      updateCounts();
    }
  );

  let activeWeights = new Float32Array(edgeCount);
  let stressedLinkCount = 0;
  function writeWeights(): void {
    const weights = new Float32Array(edgeCount);
    const replacementWeights = new Float32Array(edgeCount);
    for (let edge = 0; edge < edgeCount; edge++) {
      const edgeClass = rail.edgeClass[edge];
      const allowed =
        ctx.options.trainTypes === 'all' ||
        (ctx.options.trainTypes === 'fast' && edgeClass <= 1) ||
        (ctx.options.trainTypes === 'stopping' && edgeClass === 2);
      weights[edge] = allowed ? rail.travelTime[edge] + ctx.options.dwellSeconds : -1;
      replacementWeights[edge] = weights[edge];
    }
    activeWeights = weights;
    weightsBuffer.write(weights);
    replacementWeightsBuffer.write(replacementWeights);
    markDirty();
  }

  function writeScenario(): void {
    const corridor = CORRIDORS.find(value => value.label === ctx.options.corridor) ?? CORRIDORS[0];
    const affectedEdges = corridorEdges(rail, corridor);
    const mask = Uint32Array.from(activeWeights, weight => (weight >= 0 ? 1 : 0));
    const replacementWeights = activeWeights.slice();
    stressedLinkCount = affectedEdges.length;
    const stressedSegments = new Float32Array(Math.max(affectedEdges.length, 1) * 4);
    affectedEdges.forEach((edge, index) => {
      if (ctx.options.event === 'closure') mask[edge] = 0;
      else replacementWeights[edge] += ctx.options.slowdownMinutes * 60;
      stressedSegments.set(rail.segments.subarray(edge * 4, edge * 4 + 4), index * 4);
    });
    edgeMaskBuffer.write(mask);
    replacementWeightsBuffer.write(replacementWeights);
    stressedSegmentsBuffer.write(stressedSegments);
    markDirty();
    ctx.setReadout('corridorLinks', `${affectedEdges.length} directed edges`);
  }

  function writeOrigin(): void {
    originNode = findStation(rail, ctx.options.origin);
    startNodesBuffer.write(Uint32Array.of(originNode));
    originPositionBuffer.write(rail.nodePositions.subarray(originNode * 2, originNode * 2 + 2));
    markDirty();
    ctx.refreshTooltip();
  }

  function writeBudget(): void {
    parametersBuffer.write(
      getGPUNetworkAccessibilityParameterValues({threshold: ctx.options.budgetMinutes * 60})
    );
    markDirty();
  }

  writeWeights();
  writeScenario();
  writeOrigin();
  writeBudget();
  ctx.setReadout('stationCount', `${nodeCount} stations · ${edgeCount} directed links`);
  ctx.setReadout('solver', 'Waiting for GPU routing');

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
      if (id === 'origin') writeOrigin();
      else if (id === 'trainTypes' || id === 'dwellSeconds') {
        writeWeights();
        writeScenario();
      } else if (id === 'budgetMinutes') writeBudget();
      else if (id === 'corridor' || id === 'event' || id === 'slowdownMinutes') writeScenario();
      else ctx.requestLayers();
    },
    getLayers() {
      const dark = ctx.ground() === 'dark';
      const layers: Layer[] = [];
      if (ctx.options.showNetwork) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'rail-resilience-network',
            analysisFrame: rail.frame,
            segments: segmentBuffer,
            instanceCount: edgeCount,
            widthPixels: 1.1,
            color: dark ? [171, 187, 199, 58] : QUIET_INK
          })
        );
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'rail-resilience-stressed-link',
            analysisFrame: rail.frame,
            segments: stressedSegmentsBuffer,
            instanceCount: Math.max(stressedLinkCount, 1),
            widthPixels: 3.4,
            color: CHANGE_COLOR
          })
        );
      }
      if (summary?.converged) {
        if (ctx.options.display === 'unreachable') {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'rail-resilience-unreachable',
              analysisFrame: rail.frame,
              positions: stationBuffer,
              instanceCount: nodeCount,
              values: newlyUnreachableBuffer,
              valueFormat: 'uint32',
              colormap: 'mask',
              color: CHANGE_COLOR,
              noDataColor: dark ? [159, 185, 202, 175] : [45, 112, 149, 190],
              radiusPixels: 5.2,
              outlineColor: dark ? [10, 14, 19, 255] : [255, 255, 255, 255],
              outlineWidthPixels: 0.8
            })
          );
        } else {
          const values =
            ctx.options.display === 'baseline'
              ? baselineCostsBuffer
              : ctx.options.display === 'scenario'
                ? scenarioCostsBuffer
                : costDifferencesBuffer;
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'rail-resilience-costs',
              analysisFrame: rail.frame,
              positions: stationBuffer,
              instanceCount: nodeCount,
              values,
              valueFormat: 'float32',
              colormap: ctx.options.display === 'delta' ? 'magma' : 'viridis',
              valueScale: 1 / 60,
              valueRange: ctx.options.display === 'delta' ? [0, 30] : [0, 240],
              noDataColor: [0, 0, 0, 0],
              radiusPixels: 4.8,
              outlineColor: dark ? [10, 14, 19, 235] : [255, 255, 255, 245],
              outlineWidthPixels: 0.65
            })
          );
        }
      }
      if (ctx.options.showStations) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'rail-resilience-stations-ink',
            analysisFrame: rail.frame,
            positions: stationBuffer,
            instanceCount: nodeCount,
            radiusPixels: 1.3,
            color: dark ? [235, 242, 246, 210] : [25, 35, 44, 215]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'rail-resilience-origin',
          analysisFrame: rail.frame,
          positions: originPositionBuffer,
          instanceCount: 1,
          radiusPixels: 8,
          shape: 'ring',
          outlineWidthPixels: 1.8,
          fillOpacity: 0,
          color: [255, 119, 54, 255]
        })
      );
      return layers;
    },
    getTooltip(event) {
      const viewport = ctx.getViewport();
      if (!viewport || !summary?.converged) return null;
      const pixel = event.pixel;
      let nearest = -1;
      let nearestDistance = 14 * 14;
      for (let station = 0; station < nodeCount; station++) {
        const projected = viewport.project([
          rail.nodeLngLat[station * 2],
          rail.nodeLngLat[station * 2 + 1]
        ]);
        const distance = (projected[0] - pixel[0]) ** 2 + (projected[1] - pixel[1]) ** 2;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = station;
        }
      }
      if (nearest < 0) return null;
      const before = summary.baseline[nearest];
      const after = summary.scenario[nearest];
      const format = (value: number) =>
        Number.isFinite(value) ? `${Math.round(value / 60)} min` : 'unreachable';
      return {
        title: rail.names[nearest],
        subtitle: 'Dutch rail station',
        rows: [
          {label: 'Before', value: format(before)},
          {label: 'After', value: format(after)},
          {
            label: 'Change',
            value: Number.isFinite(summary.delta[nearest])
              ? `${Math.round(summary.delta[nearest] / 60)} min`
              : summary.unreachable[nearest]
                ? 'newly unreachable'
                : 'not comparable',
            swatch: CHANGE_COLOR
          }
        ],
        note: 'Scheduled median in-vehicle time; perfect transfers, no waiting.'
      };
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
