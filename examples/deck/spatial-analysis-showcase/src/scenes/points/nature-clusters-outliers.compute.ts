// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {
  getGPULocalOutlierFactorParameterValues,
  getGPUNeighborSearchParameterValues,
  GPULocalOutlierFactor,
  GPUNeighborSearch
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {type GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {formatCount} from '../../cartography/live-text';
import {DENSE_POINT_RADIUS_STOPS} from '../../cartography/zoom';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {createAnalysisFrameProjection, projectLngLatToFrame} from '../../engine/analysis-frame';
import {GraphSwap} from '../../engine/graph-swap';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {readNatureColumns} from './b1-nature-data';
import {getSubjectColor} from './b1-points-look';
import {
  makeLocalOutlierFactorHistogram,
  NATURE_OUTLIER_SAMPLE_LIMIT,
  selectEvenlySpacedRows
} from './nature-clusters-outliers-analysis';

const NEIGHBOR_GRID_SIZE: readonly [number, number] = [32, 32];
const HISTOGRAM_BIN_COUNT = 24;
const HISTOGRAM_MAXIMUM_SCORE = 4;
const DENSITY_FLOOR_METERS = 10;

/** Option state for the companion local-sparsity scene. */
export type NatureOutlierOptions = {
  category: string;
  view: 'records' | 'score';
  neighborCount: number;
  threshold: number;
};

type OutlierGraph = {
  resources: SpatialAnalysisResources;
  graph: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  lofParameters: GPUParameterBuffer<'float32'>;
  positions: Buffer;
  scores: Buffer;
  pointCount: number;
  category: string;
  neighborCount: number;
  threshold: number;
  readbackState: {threshold: number};
  dirty: boolean;
};

/** Runs LOF over a deterministic bounded sample and draws the continuous GPU score. */
export async function createNatureOutliers(
  ctx: SceneContext<NatureOutlierOptions>
): Promise<SceneInstance<NatureOutlierOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const columns = readNatureColumns(observations);
  const origin = columns.origin;
  const projection = createAnalysisFrameProjection({kind: 'azimuthal-equidistant', origin});
  const projected = projectLngLatToFrame(projection, observations.column<Float32Array>('position'));
  const swapResources = new SpatialAnalysisResources(device, 'nature-lof-swap');
  const graphSwap = swapResources.track(new GraphSwap());
  let serial = 0;
  let destroyed = false;
  let active: OutlierGraph;

  const getSourceRows = (category: string): Uint32Array => {
    const categoryIndex = columns.categoryNames.indexOf(category);
    const matchingRows: number[] = [];
    for (let row = 0; row < columns.count; row++) {
      if (columns.category[row] === categoryIndex) matchingRows.push(row);
    }
    return selectEvenlySpacedRows(Uint32Array.from(matchingRows), NATURE_OUTLIER_SAMPLE_LIMIT);
  };

  const buildGraph = async (options: NatureOutlierOptions): Promise<OutlierGraph> => {
    const buildOptions = {...options};
    const instanceSerial = ++serial;
    const sourceRows = getSourceRows(buildOptions.category);
    if (sourceRows.length < 2)
      throw new Error('This group has too few records for local comparison.');
    const pointCount = sourceRows.length;
    const positions = new Float32Array(pointCount * 2);
    for (let row = 0; row < pointCount; row++) {
      const sourceRow = sourceRows[row];
      positions[row * 2] = projected[sourceRow * 2];
      positions[row * 2 + 1] = projected[sourceRow * 2 + 1];
    }
    let minimumX = Infinity;
    let maximumX = -Infinity;
    let minimumY = Infinity;
    let maximumY = -Infinity;
    for (let row = 0; row < pointCount; row++) {
      minimumX = Math.min(minimumX, positions[row * 2]);
      maximumX = Math.max(maximumX, positions[row * 2]);
      minimumY = Math.min(minimumY, positions[row * 2 + 1]);
      maximumY = Math.max(maximumY, positions[row * 2 + 1]);
    }
    const padding = 1;
    const bounds: [number, number, number, number] = [
      minimumX - padding,
      minimumY - padding,
      maximumX + padding,
      maximumY + padding
    ];
    const neighborCount = Math.min(Math.max(2, Math.round(buildOptions.neighborCount)), pointCount);
    const resources = new SpatialAnalysisResources(device, `nature-lof-${instanceSerial}`);
    const positionsBuffer = resources.createBuffer('positions', positions);
    const scores = resources.createBuffer('scores', pointCount * 4);
    const parameterBuffer = resources.createParameterBuffer(
      'neighbor-parameters',
      'float32',
      12,
      getGPUNeighborSearchParameterValues({bounds, spatialContext: projection.spatialContext})
    );
    const lofParameters = resources.createParameterBuffer(
      'lof-parameters',
      'float32',
      4,
      getGPULocalOutlierFactorParameterValues({
        threshold: buildOptions.threshold,
        densityFloor: DENSITY_FLOOR_METERS
      })
    );
    const graph = new GPUCommandGraph<void>(device, {id: `nature-lof-${instanceSerial}`});
    const graphPositions = importGraphBuffer(
      graph,
      'positions',
      positionsBuffer,
      'float32x2',
      pointCount
    );
    const makeView = (name: string, length: number) => resources.createBuffer(name, length * 4);
    const neighborOffsets = makeView('neighbor-offsets', pointCount + 1);
    const neighborIds = makeView('neighbor-ids', pointCount * neighborCount);
    const neighborWeights = makeView('neighbor-weights', pointCount * neighborCount);
    const neighborDistances = makeView('neighbor-distances', pointCount * neighborCount);
    const neighborOverflow = makeView('neighbor-overflow', 1);
    const localReachabilityDensity = makeView('local-density', pointCount);
    const kDistance = makeView('k-distance', pointCount);
    const outlier = makeView('outlier-mask', pointCount);
    const outlierCount = makeView('outlier-count', 1);
    const search = new GPUNeighborSearch({
      id: 'nature-nearest-records',
      spatialContext: projection.spatialContext,
      mode: 'knn',
      neighborCount,
      positions: graphPositions,
      parameters: parameterBuffer.importToGraph(graph),
      gridSize: NEIGHBOR_GRID_SIZE,
      output: {
        offsets: importGraphBuffer(
          graph,
          'neighborOffsets',
          neighborOffsets,
          'uint32',
          pointCount + 1
        ),
        neighbors: importGraphBuffer(
          graph,
          'neighborIds',
          neighborIds,
          'uint32',
          pointCount * neighborCount
        ),
        weights: importGraphBuffer(
          graph,
          'neighborWeights',
          neighborWeights,
          'float32',
          pointCount * neighborCount
        ),
        distances: importGraphBuffer(
          graph,
          'neighborDistances',
          neighborDistances,
          'float32',
          pointCount * neighborCount
        ),
        status: {
          overflow: importGraphBuffer(graph, 'neighborOverflow', neighborOverflow, 'uint32', 1)
        }
      }
    });
    graph.add(search);
    graph.add(
      new GPULocalOutlierFactor({
        id: 'nature-local-outlier-factor',
        spatialContext: projection.spatialContext,
        neighbors: search.outputWeights,
        parameters: lofParameters.importToGraph(graph),
        output: {
          kDistance: importGraphBuffer(graph, 'kDistance', kDistance, 'float32', pointCount),
          localReachabilityDensity: importGraphBuffer(
            graph,
            'localDensity',
            localReachabilityDensity,
            'float32',
            pointCount
          ),
          lof: importGraphBuffer(graph, 'lof', scores, 'float32', pointCount),
          outlier: importGraphBuffer(graph, 'outlier', outlier, 'uint32', pointCount),
          outlierCount: importGraphBuffer(graph, 'outlierCount', outlierCount, 'uint32', 1)
        }
      })
    );
    try {
      const compiled = resources.track(await graph.compileAsync());
      const readbackState = {threshold: buildOptions.threshold};
      const reader = new SummaryReader(
        resources,
        `nature-lof-${instanceSerial}`,
        [
          {buffer: scores, size: pointCount * 4},
          {buffer: outlierCount, size: 4},
          {buffer: neighborOverflow, size: 4}
        ],
        bytes => {
          if (destroyed || active?.resources !== resources) return;
          const scoreValues = new Float32Array(bytes, 0, pointCount).slice();
          const summaryOffset = pointCount * 4;
          const flaggedCount = new Uint32Array(bytes, summaryOffset, 1)[0];
          const overflow = new Uint32Array(bytes, summaryOffset + 4, 1)[0];
          const sortedScores = [...scoreValues]
            .filter(Number.isFinite)
            .sort((first, second) => first - second);
          const medianIndex = Math.floor(sortedScores.length / 2);
          const medianScore = sortedScores.length
            ? sortedScores.length % 2 === 0
              ? (sortedScores[medianIndex - 1] + sortedScores[medianIndex]) / 2
              : sortedScores[medianIndex]
            : 0;
          const maximumScore = sortedScores.at(-1) ?? 0;
          ctx.setReadout('sampleSize', formatCount(pointCount));
          ctx.setReadout(
            'flagged',
            readbackState.threshold === ctx.options.threshold
              ? `${formatCount(flaggedCount)} of ${formatCount(pointCount)}`
              : 'Updating for the new cut…'
          );
          ctx.setReadout('medianScore', medianScore.toFixed(2));
          ctx.setReadout('maximumScore', maximumScore.toFixed(2));
          ctx.setReadout(
            'neighborOverflow',
            overflow ? 'kNN output truncated by capacity' : 'None'
          );
          ctx.setChart('scoreDistribution', {
            kind: 'histogram',
            values: makeLocalOutlierFactorHistogram(
              scoreValues,
              HISTOGRAM_BIN_COUNT,
              HISTOGRAM_MAXIMUM_SCORE
            ),
            xDomain: [0, HISTOGRAM_MAXIMUM_SCORE],
            title: 'Sampled records by LOF score',
            xLabel: 'Local outlier factor',
            yLabel: 'Records',
            formatX: value => value.toFixed(1),
            now: ctx.options.threshold,
            nowLabel: 'Chosen cut',
            table: false,
            description: 'Continuous GPU scores; scores at or above 4 are grouped in the final bin.'
          });
          ctx.requestLayers();
        }
      );
      return {
        resources,
        graph: compiled,
        reader,
        lofParameters,
        positions: positionsBuffer,
        scores,
        pointCount,
        category: buildOptions.category,
        neighborCount,
        threshold: buildOptions.threshold,
        readbackState,
        dirty: true
      };
    } catch (error) {
      resources.destroy();
      throw error;
    }
  };

  const publishData = (set: OutlierGraph) => {
    ctx.setReadout('sampleSize', formatCount(set.pointCount));
    ctx.setReadout('categoryLabel', set.category);
    ctx.setReadout('neighborCount', formatCount(set.neighborCount));
    ctx.setReadout('flagged', null);
    ctx.setReadout('medianScore', null);
    ctx.setReadout('maximumScore', null);
    ctx.setReadout('neighborOverflow', null);
    ctx.setChart('scoreDistribution', null);
  };

  active = await buildGraph(ctx.options);
  publishData(active);

  const scheduleRebuild = () => {
    ctx.setStatus('Rebuilding the bounded k-nearest-neighbour graph…');
    void graphSwap
      .request({
        build: () => buildGraph({...ctx.options}),
        commit(next) {
          next.threshold = ctx.options.threshold;
          next.readbackState.threshold = next.threshold;
          next.lofParameters.write(
            getGPULocalOutlierFactorParameterValues({
              threshold: next.threshold,
              densityFloor: DENSITY_FLOOR_METERS
            })
          );
          const previous = active;
          active = next;
          previous.reader.stop();
          previous.resources.destroy();
          publishData(next);
          ctx.setStatus('');
          ctx.requestLayers();
        },
        discard(next) {
          next.reader.stop();
          next.resources.destroy();
        }
      })
      .catch(error => {
        if (!destroyed) ctx.setStatus(`Could not rebuild: ${String(error)}`);
      });
  };

  return {
    getCompiledGraphs: () => [active.graph],
    setOption(id, _value, state) {
      if (id === 'category' || id === 'neighborCount') {
        scheduleRebuild();
      } else if (id === 'view') {
        ctx.requestLayers();
      } else if (id === 'threshold') {
        active.threshold = state.threshold;
        active.lofParameters.write(
          getGPULocalOutlierFactorParameterValues({
            threshold: state.threshold,
            densityFloor: DENSITY_FLOOR_METERS
          })
        );
        active.dirty = true;
      }
    },
    onGroundChange() {
      ctx.requestLayers();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    encode(commandEncoder) {
      if (active.dirty) {
        active.dirty = false;
        active.graph.encode(commandEncoder, {parameters: undefined});
        active.reader.markStale();
      }
      if (!active.reader.isPending) {
        active.readbackState.threshold = active.threshold;
      }
      active.reader.flush(commandEncoder);
    },
    getLayers() {
      if (ctx.options.view === 'records') {
        return [
          new SpatialAnalysisPointLayer({
            id: `nature-lof-records-${serial}`,
            coordinateOrigin: [origin[0], origin[1], 0],
            analysisFrame: projection.frame,
            positions: active.positions,
            instanceCount: active.pointCount,
            colormap: 'uniform',
            color: getSubjectColor(ctx.ground(), 170),
            radiusPixels: DENSE_POINT_RADIUS_STOPS
          })
        ];
      }
      return [
        new SpatialAnalysisPointLayer({
          id: `nature-lof-points-${serial}`,
          coordinateOrigin: [origin[0], origin[1], 0],
          analysisFrame: projection.frame,
          positions: active.positions,
          instanceCount: active.pointCount,
          values: active.scores,
          valueFormat: 'float32',
          valueRange: [0, HISTOGRAM_MAXIMUM_SCORE],
          colormap: 'oranges',
          radiusPixels: DENSE_POINT_RADIUS_STOPS
        })
      ] as Layer[];
    },
    destroy() {
      destroyed = true;
      active.reader.stop();
      active.resources.destroy();
      swapResources.destroy();
    }
  };
}
