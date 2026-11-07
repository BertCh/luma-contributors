// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUConditionalGaussianSimulation} from '@luma-contributors/graph/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {MEUSE_LOG_RESPONSE_RANGES} from './meuse-validation.constants';
import {
  MEUSE_SIMULATION_QUERY_ROWS,
  MEUSE_SIMULATION_SAMPLE_ROWS
} from './meuse-conditional-simulation.selection';
import type {SceneContext, SceneInstance, ScenePointerEvent} from '../scene';

export type MeuseConditionalSimulationOptions = {
  knownMean: '5.5' | '5.9' | '6.3';
  observationErrorVariance: '0' | '0.01' | '0.04' | '0.16';
  seed: number;
  replicateCount: '32' | '128' | '256';
  replicateOffset: '0' | '256' | '512';
  replicateIndex: number;
  display: 'realization' | 'mean' | 'variance';
};

const MAXIMUM_REPLICATE_COUNT = 256;
const QUERY_COUNT = MEUSE_SIMULATION_QUERY_ROWS.length;
const SAMPLE_COUNT = MEUSE_SIMULATION_SAMPLE_ROWS.length;
const DISPLAY_REALIZATION = 0;
const DISPLAY_MEAN = 1;
const DISPLAY_VARIANCE = 2;

function getSimulationDisplayIndex(display: MeuseConditionalSimulationOptions['display']): number {
  return display === 'mean'
    ? DISPLAY_MEAN
    : display === 'variance'
      ? DISPLAY_VARIANCE
      : DISPLAY_REALIZATION;
}

/** Bounded exact joint conditional-Gaussian simulations on a reproducible Meuse subset. */
export async function createMeuseConditionalSimulation(
  ctx: SceneContext<MeuseConditionalSimulationOptions>
): Promise<SceneInstance<MeuseConditionalSimulationOptions>> {
  const dataset = ctx.datasets.get('meuse-validation');
  const {device} = ctx;
  const origin = dataset.defaultOrigin;
  const projection = dataset.getProjection(origin);
  const sampleSource = dataset.column('samplePositionRD');
  const querySource = dataset.column('gridPositionRD');
  const samplePositionSource = new Float32Array(SAMPLE_COUNT * 2);
  const queryPositionSource = new Float32Array(QUERY_COUNT * 2);
  const sampleValues = new Float32Array(SAMPLE_COUNT);
  const observations = new Float32Array(SAMPLE_COUNT * 4);
  const sampleMapPositions = new Float32Array(SAMPLE_COUNT * 2);
  const queryMapPositions = new Float32Array(QUERY_COUNT * 2);
  const allSampleMapPositions = dataset.projectColumn('samplePosition', origin);
  const allQueryMapPositions = dataset.projectColumn('gridPosition', origin);
  const logZinc = dataset.column('logZinc');
  let minimumX = Infinity;
  let maximumX = -Infinity;
  let minimumY = Infinity;
  let maximumY = -Infinity;
  for (let row = 0; row < dataset.column('sampleDist').length; row++) {
    minimumX = Math.min(minimumX, sampleSource[row * 2]);
    maximumX = Math.max(maximumX, sampleSource[row * 2]);
    minimumY = Math.min(minimumY, sampleSource[row * 2 + 1]);
    maximumY = Math.max(maximumY, sampleSource[row * 2 + 1]);
  }
  const centerX = (minimumX + maximumX) / 2;
  const centerY = (minimumY + maximumY) / 2;
  for (let index = 0; index < SAMPLE_COUNT; index++) {
    const sourceRow = MEUSE_SIMULATION_SAMPLE_ROWS[index];
    const coordinateX = sampleSource[sourceRow * 2];
    const coordinateY = sampleSource[sourceRow * 2 + 1];
    samplePositionSource.set([coordinateX - centerX, coordinateY - centerY], index * 2);
    sampleValues[index] = logZinc[sourceRow];
    observations.set(
      [sampleValues[index], Number(ctx.options.observationErrorVariance), 0, 1],
      index * 4
    );
    sampleMapPositions.set(
      allSampleMapPositions.subarray(sourceRow * 2, sourceRow * 2 + 2),
      index * 2
    );
  }
  for (let index = 0; index < QUERY_COUNT; index++) {
    const sourceRow = MEUSE_SIMULATION_QUERY_ROWS[index];
    const coordinateX = querySource[sourceRow * 2];
    const coordinateY = querySource[sourceRow * 2 + 1];
    queryPositionSource.set([coordinateX - centerX, coordinateY - centerY], index * 2);
    queryMapPositions.set(
      allQueryMapPositions.subarray(sourceRow * 2, sourceRow * 2 + 2),
      index * 2
    );
  }
  const riverCoordinates =
    (dataset.geojson?.features[0]?.geometry as {coordinates: [number, number][]} | undefined)
      ?.coordinates ?? [];
  const riverSegments = new Float32Array(Math.max(0, riverCoordinates.length - 1) * 4);
  for (let index = 0; index + 1 < riverCoordinates.length; index++) {
    const first = projection.project(...riverCoordinates[index]);
    const second = projection.project(...riverCoordinates[index + 1]);
    riverSegments.set([first[0], first[1], second[0], second[1]], index * 4);
  }

  const resources = new SpatialAnalysisResources(device, 'meuse-conditional-simulation');
  const positionsBuffer = resources.createBuffer(
    'simulation-sample-positions',
    samplePositionSource
  );
  const observationsBuffer = resources.createBuffer('simulation-observations', observations);
  const queriesBuffer = resources.createBuffer('simulation-query-positions', queryPositionSource);
  const drawValuesBuffer = resources.createBuffer(
    'simulation-draw-values',
    QUERY_COUNT * MAXIMUM_REPLICATE_COUNT * 4
  );
  const momentsBuffer = resources.createBuffer('simulation-conditional-moments', QUERY_COUNT * 8);
  const statusBuffer = resources.createBuffer('simulation-status', MAXIMUM_REPLICATE_COUNT * 4);
  const selectedMapValues = resources.createBuffer(
    'simulation-selected-map-values',
    QUERY_COUNT * 4
  );
  const mapControlBuffer = resources.createBuffer(
    'simulation-map-control',
    Uint32Array.of(0, 32, 0)
  );
  const generationInput = resources.createBuffer('simulation-generation-input', Uint32Array.of(0));
  const generationOutput = resources.createBuffer('simulation-generation-output', 4);
  const sampleMapBuffer = resources.createBuffer(
    'simulation-sample-map-positions',
    sampleMapPositions,
    Buffer.VERTEX
  );
  const queryMapBuffer = resources.createBuffer(
    'simulation-query-map-positions',
    queryMapPositions,
    Buffer.VERTEX
  );
  const observedMapValues = resources.createBuffer(
    'simulation-observed-log-zinc',
    sampleValues,
    Buffer.VERTEX
  );
  const riverBuffer = resources.createBuffer('simulation-river-outline', riverSegments);
  const graph = new GPUCommandGraph<void>(device, {id: 'meuse-conditional-simulation'});
  const positionView = importGraphBuffer(
    graph,
    'simulation-sample-positions',
    positionsBuffer,
    'float32x2',
    SAMPLE_COUNT
  );
  const observationsView = importGraphBuffer(
    graph,
    'simulation-observations',
    observationsBuffer,
    'float32x4',
    SAMPLE_COUNT
  );
  const queryView = importGraphBuffer(
    graph,
    'simulation-query-positions',
    queriesBuffer,
    'float32x2',
    QUERY_COUNT
  );
  const drawValuesView = importGraphBuffer(
    graph,
    'simulation-draw-values',
    drawValuesBuffer,
    'float32',
    QUERY_COUNT * MAXIMUM_REPLICATE_COUNT
  );
  const momentsView = importGraphBuffer(
    graph,
    'simulation-conditional-moments',
    momentsBuffer,
    'float32x2',
    QUERY_COUNT
  );
  const statusView = importGraphBuffer(
    graph,
    'simulation-status',
    statusBuffer,
    'uint32',
    MAXIMUM_REPLICATE_COUNT
  );
  const selectedMapValuesView = importGraphBuffer(
    graph,
    'simulation-selected-map-values',
    selectedMapValues,
    'float32',
    QUERY_COUNT
  );
  const mapControlView = importGraphBuffer(
    graph,
    'simulation-map-control',
    mapControlBuffer,
    'uint32',
    3
  );
  const generationInputView = importGraphBuffer(
    graph,
    'simulation-generation-input',
    generationInput,
    'uint32',
    1
  );
  const generationOutputView = importGraphBuffer(
    graph,
    'simulation-generation-output',
    generationOutput,
    'uint32',
    1
  );
  const simulation = new GPUConditionalGaussianSimulation({
    id: 'meuse-conditional-gaussian',
    positions: positionView,
    observations: observationsView,
    queries: queryView,
    covariance: {model: 'spherical', range: 874, sill: 0.59, nugget: 0.04},
    mean: Number(ctx.options.knownMean),
    seed: ctx.options.seed,
    replicateCount: MAXIMUM_REPLICATE_COUNT,
    reusePreparation: true,
    output: {values: drawValuesView, moments: momentsView, status: statusView}
  });
  graph.add(simulation);
  simulation.setDrawParameters({
    seed: ctx.options.seed,
    replicateCount: Number(ctx.options.replicateCount)
  });
  addKernelPass(graph, {
    id: 'meuse-simulation-select-map-values',
    invocationCount: QUERY_COUNT,
    declarations:
      'fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }',
    bindings: [
      {name: 'draws', view: drawValuesView, type: 'f32', access: 'read'},
      {name: 'moments', view: momentsView, type: 'f32', access: 'read'},
      {name: 'statuses', view: statusView, type: 'u32', access: 'read'},
      {name: 'control', view: mapControlView, type: 'u32', access: 'read'},
      {name: 'values', view: selectedMapValuesView, type: 'f32', access: 'read_write'}
    ],
    body: `let display = control[controlOffset]; let replicateCount = control[controlOffset + 1u]; let replicate = control[controlOffset + 2u]; let isActive = replicateCount > 0u && statuses[statusesOffset] == 0u; let mean = moments[momentsOffset + index * 2u]; let variance = moments[momentsOffset + index * 2u + 1u]; var value = bitcast<f32>(0x7fc00000u | (index & 0u)); if (isActive && display == ${DISPLAY_MEAN}u && finite(mean)) { value = mean; } else if (isActive && display == ${DISPLAY_VARIANCE}u && finite(variance)) { value = variance; } else if (isActive && display == ${DISPLAY_REALIZATION}u && replicate < replicateCount && statuses[statusesOffset + replicate] == 0u) { value = draws[drawsOffset + replicate * ${QUERY_COUNT}u + index]; } values[valuesOffset + index] = value;`
  });
  addKernelPass(graph, {
    id: 'meuse-simulation-generation-snapshot',
    invocationCount: 1,
    bindings: [
      {name: 'generation', view: generationInputView, type: 'u32', access: 'read'},
      {name: 'snapshot', view: generationOutputView, type: 'u32', access: 'read_write'}
    ],
    body: 'snapshot[snapshotOffset] = generation[generationOffset];'
  });
  const compiled = resources.track(await graph.compileAsync());
  const mapReader = new SummaryReader(
    resources,
    'meuse-conditional-simulation-summary',
    [
      {buffer: momentsBuffer, size: QUERY_COUNT * 8},
      {buffer: drawValuesBuffer, size: QUERY_COUNT * MAXIMUM_REPLICATE_COUNT * 4},
      {buffer: statusBuffer, size: MAXIMUM_REPLICATE_COUNT * 4},
      {buffer: generationOutput, size: 4}
    ],
    bytes => {
      const generation = new Uint32Array(
        bytes,
        QUERY_COUNT * 8 + QUERY_COUNT * MAXIMUM_REPLICATE_COUNT * 4 + MAXIMUM_REPLICATE_COUNT * 4,
        1
      )[0];
      if (generation !== currentGeneration) return;
      const moments = new Float32Array(bytes, 0, QUERY_COUNT * 2);
      const drawOffset = QUERY_COUNT * 8;
      const draws = new Float32Array(bytes, drawOffset, QUERY_COUNT * MAXIMUM_REPLICATE_COUNT);
      const statusOffset = drawOffset + QUERY_COUNT * MAXIMUM_REPLICATE_COUNT * 4;
      const statuses = new Uint32Array(bytes, statusOffset, MAXIMUM_REPLICATE_COUNT);
      latestMoments = moments;
      latestDraws = draws;
      latestStatuses = statuses;
      const requestedCount = Number(ctx.options.replicateCount);
      const successfulCount = Array.from(statuses.slice(0, requestedCount)).filter(
        value => value === 0
      ).length;
      if (successfulCount !== requestedCount) {
        latestMoments = null;
        latestDraws = null;
        latestStatuses = null;
        ctx.setReadout(
          'simulationStatus',
          `${successfulCount}/${requestedCount} joint draws succeeded; numerical failures are reported by the GPU status output.`
        );
        ctx.setChart('meanComparison', null);
        ctx.setChart('varianceComparison', null);
        ctx.requestLayers();
        return;
      }
      let analyticMean = 0;
      let analyticVariance = 0;
      let empiricalMean = 0;
      let empiricalVariance = 0;
      for (let query = 0; query < QUERY_COUNT; query++) {
        const expectedMean = moments[query * 2];
        const expectedVariance = moments[query * 2 + 1];
        let drawMean = 0;
        for (let replicate = 0; replicate < requestedCount; replicate++) {
          drawMean += draws[replicate * QUERY_COUNT + query] / requestedCount;
        }
        let drawVariance = 0;
        for (let replicate = 0; replicate < requestedCount; replicate++) {
          const difference = draws[replicate * QUERY_COUNT + query] - drawMean;
          drawVariance += (difference * difference) / Math.max(1, requestedCount - 1);
        }
        analyticMean += expectedMean / QUERY_COUNT;
        analyticVariance += expectedVariance / QUERY_COUNT;
        empiricalMean += drawMean / QUERY_COUNT;
        empiricalVariance += drawVariance / QUERY_COUNT;
      }
      ctx.setReadout(
        'simulationStatus',
        `${successfulCount} joint draws succeeded · seed ${ctx.options.seed} · replicate IDs ${ctx.options.replicateOffset}–${Number(ctx.options.replicateOffset) + requestedCount - 1} · mean ${ctx.options.knownMean} log mg/kg · observation-error variance ${ctx.options.observationErrorVariance} log units² · 128 conditioned samples / 128 query locations`
      );
      ctx.setChart('meanComparison', {
        kind: 'bars',
        title: 'Conditional mean · analytic vs draw average',
        labels: ['Analytic mean', 'Average of draws'],
        values: [analyticMean, empiricalMean],
        description:
          'Values are averaged across the 128 selected query locations. Monte Carlo uncertainty in this average depends on the full conditional covariance; the spatially correlated query locations are not independent replicates.'
      });
      ctx.setChart('varianceComparison', {
        kind: 'bars',
        title: 'Conditional variance · analytic vs draw variance',
        labels: ['Analytic variance', 'Variance of draws'],
        values: [analyticVariance, empiricalVariance],
        description:
          'Unbiased sample variance across requested joint draws is averaged across the 128 query locations; this is a Monte Carlo approximation whose across-query uncertainty depends on the full conditional covariance, not 128 independent locations.'
      });
      ctx.setReadout(
        'momentComparison',
        `Across-query mean ${analyticMean.toFixed(3)} analytic / ${empiricalMean.toFixed(3)} simulated · variance ${analyticVariance.toFixed(3)} analytic / ${empiricalVariance.toFixed(3)} simulated (unbiased, log units²)`
      );
      ctx.requestLayers();
    }
  );

  let graphDirty = true;
  let currentGeneration = 0;
  let latestDraws: Float32Array | null = null;
  let latestMoments: Float32Array | null = null;
  let latestStatuses: Uint32Array | null = null;
  const currentOptions = ctx.options;
  mapControlBuffer.write(
    Uint32Array.of(
      getSimulationDisplayIndex(currentOptions.display),
      Number(currentOptions.replicateCount),
      currentOptions.replicateIndex
    )
  );
  ctx.setReadout('sampleCount', SAMPLE_COUNT);
  ctx.setReadout('queryCount', QUERY_COUNT);
  ctx.setReadout(
    'selectionDesign',
    'Maximin sample subset (source rows 1–154) · 128 nearest grid points to its in-hull query centroid'
  );
  ctx.setReadout(
    'simulationStatus',
    'Conditional Gaussian preparation and first draw batch pending.'
  );
  ctx.setReadout('momentComparison', 'Analytic and empirical moments pending GPU execution.');
  ctx.setChart('meanComparison', null);
  ctx.setChart('varianceComparison', null);
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (graphDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        mapReader.request(commandEncoder);
        graphDirty = false;
      } else {
        mapReader.flush(commandEncoder);
      }
    },
    setOption(id, _value, state) {
      currentGeneration++;
      generationInput.write(Uint32Array.of(currentGeneration));
      mapReader.markStale();
      if (id === 'knownMean') {
        simulation.setMean(Number(state.knownMean));
      } else if (id === 'replicateCount' || id === 'replicateOffset' || id === 'seed') {
        simulation.setDrawParameters({
          seed: state.seed,
          replicateOffset: Number(state.replicateOffset),
          replicateCount: Number(state.replicateCount)
        });
      } else if (id === 'observationErrorVariance') {
        const errorVariance = Number(state.observationErrorVariance);
        for (let sample = 0; sample < SAMPLE_COUNT; sample++) {
          observations[sample * 4 + 1] = errorVariance;
        }
        observationsBuffer.write(observations);
        simulation.invalidatePreparation();
      }
      mapControlBuffer.write(
        Uint32Array.of(
          getSimulationDisplayIndex(state.display),
          Number(state.replicateCount),
          state.replicateIndex
        )
      );
      graphDirty = true;
      latestMoments = null;
      latestDraws = null;
      latestStatuses = null;
      ctx.setReadout('simulationStatus', 'Refreshing the requested joint draw batch.');
      if (id === 'display') ctx.refreshLegends();
      ctx.requestLayers();
      ctx.refreshTooltip();
    },
    getLayers(): Layer[] {
      const layers: Layer[] = [];
      if (latestMoments && latestDraws && latestStatuses) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'meuse-conditional-simulation-query-map',
            coordinateOrigin: [origin[0], origin[1], 0],
            positions: queryMapBuffer,
            values: selectedMapValues,
            valueFormat: 'float32',
            instanceCount: QUERY_COUNT,
            radiusPixels: 5,
            colormap: 'cividis',
            valueRange:
              ctx.options.display === 'variance' ? [0, 0.6] : MEUSE_LOG_RESPONSE_RANGES.zinc,
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'meuse-conditional-simulation-river',
          coordinateOrigin: [origin[0], origin[1], 0],
          segments: riverBuffer,
          instanceCount: riverSegments.length / 4,
          widthPixels: 2,
          color: [49, 113, 151, 230]
        })
      );
      if (ctx.options.display !== 'variance') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'meuse-conditional-simulation-samples',
            coordinateOrigin: [origin[0], origin[1], 0],
            positions: sampleMapBuffer,
            values: observedMapValues,
            valueFormat: 'float32',
            instanceCount: SAMPLE_COUNT,
            radiusPixels: 4.5,
            colormap: 'cividis',
            valueRange: MEUSE_LOG_RESPONSE_RANGES.zinc,
            outlineColor: [16, 18, 22, 255],
            outlineWidthPixels: 1
          })
        );
      }
      return layers;
    },
    getTooltip(event: ScenePointerEvent): string | null {
      if (!event.coordinate) return null;
      const projected = projection.project(event.coordinate[0], event.coordinate[1]);
      let nearest = -1;
      let distance = Infinity;
      for (let index = 0; index < QUERY_COUNT; index++) {
        const candidateDistance = Math.hypot(
          queryMapPositions[index * 2] - projected[0],
          queryMapPositions[index * 2 + 1] - projected[1]
        );
        if (candidateDistance < distance) {
          nearest = index;
          distance = candidateDistance;
        }
      }
      if (nearest < 0 || distance > 35 || !latestMoments || !latestDraws || !latestStatuses) {
        return null;
      }
      const mean = latestMoments[nearest * 2];
      const variance = latestMoments[nearest * 2 + 1];
      const replicate = ctx.options.replicateIndex;
      const lines = [
        `Conditional log-Zn at selected query ${MEUSE_SIMULATION_QUERY_ROWS[nearest]}`,
        `Analytic mean ${mean.toFixed(3)} log mg/kg · variance ${variance.toFixed(3)} log units²`
      ];
      if (replicate < Number(ctx.options.replicateCount) && latestStatuses[replicate] === 0) {
        const realization = latestDraws[replicate * QUERY_COUNT + nearest];
        lines.push(
          `Replicate ${replicate} simulated concentration ${Math.exp(realization).toFixed(2)} mg/kg`
        );
      }
      lines.push('Observed sample subset: 128 of the 155 source sites');
      return lines.join('\n');
    },
    destroy() {
      mapReader.stop();
      resources.release(compiled);
      simulation.destroy();
      resources.destroy();
    }
  };
}
