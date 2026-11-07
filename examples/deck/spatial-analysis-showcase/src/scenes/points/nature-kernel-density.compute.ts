// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity,
  type GPUKernelDensityBandwidthRule,
  type GPUKernelDensityEdgeCorrection,
  type GPUKernelDensityKernel,
  type GPUKernelDensityMethod,
  type GPUKernelDensityNormalization
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';

export type NatureKernelDensityOptions = {
  kernel: GPUKernelDensityKernel;
  edgeCorrection: GPUKernelDensityEdgeCorrection;
  normalization: GPUKernelDensityNormalization;
  bandwidth: number;
  bandwidthRule: 'fixed' | GPUKernelDensityBandwidthRule;
  method: GPUKernelDensityMethod;
  weight: 'observations' | 'researchGrade';
  showSurface: boolean;
  ramp: 'magma' | 'inferno' | 'fire' | 'mako';
};

const WIDTH = 64;
const HEIGHT = 64;
const CELL_SIZE = 250;
const ANALYSIS_FRAME = {kind: 'lambert-equal-area', origin: [-87.68, 41.88]} as const;
const FRAME = createAnalysisFrameProjection(ANALYSIS_FRAME);
const EXTENT = [-8000, -8000, 8000, 8000] as const;
const GRID = makeGPURasterMetadata({
  width: WIDTH,
  height: HEIGHT,
  cellSize: CELL_SIZE,
  origin: [-8000, 8000]
});
const CELL_COUNT = WIDTH * HEIGHT;
const INDEX_GRID_SIZE = [32, 32] as const;
const EMPTY_EXTENT: readonly [number, number] = [0, 1];
const WINDOW_SEGMENTS = new Float32Array([
  EXTENT[0],
  EXTENT[1],
  EXTENT[2],
  EXTENT[1],
  EXTENT[2],
  EXTENT[1],
  EXTENT[2],
  EXTENT[3],
  EXTENT[2],
  EXTENT[3],
  EXTENT[0],
  EXTENT[3],
  EXTENT[0],
  EXTENT[3],
  EXTENT[0],
  EXTENT[1]
]);

type DensityGraph = {
  graph: CompiledGPUCommandGraph<void>;
  density: ReturnType<SpatialAnalysisResources['createBuffer']>;
  summary: SummaryReader;
};

export async function createNatureKernelDensity(
  ctx: SceneContext<NatureKernelDensityOptions>
): Promise<SceneInstance<NatureKernelDensityOptions>> {
  const dataset = ctx.datasets.get('chicago-nature');
  const positions = dataset.column<Float32Array>('position');
  const projected: number[] = [];
  const sourceIndices: number[] = [];
  const researchGrade = dataset.column<Uint8Array>('researchGrade');
  for (let index = 0; index < positions.length / 2; index++) {
    const [projectedEasting, projectedNorthing] = FRAME.project(
      positions[index * 2],
      positions[index * 2 + 1]
    );
    if (
      projectedEasting >= EXTENT[0] &&
      projectedEasting <= EXTENT[2] &&
      projectedNorthing >= EXTENT[1] &&
      projectedNorthing <= EXTENT[3]
    ) {
      projected.push(projectedEasting, projectedNorthing);
      sourceIndices.push(index);
    }
  }
  const pointCount = sourceIndices.length;
  const resources = new SpatialAnalysisResources(ctx.device, 'nature-kernel-density');
  const positionBuffer = resources.createBuffer('positions', Float32Array.from(projected));
  const weightValues = new Float32Array(pointCount);
  const weightBuffer = resources.createBuffer('weights', weightValues);
  const densityBuffer = resources.createBuffer('density', CELL_COUNT * 4);
  const windowSegments = resources.createBuffer('window-segments', WINDOW_SEGMENTS);
  const bandwidthBuffer = resources.createBuffer('bandwidth', 4);
  const invalidCountBuffer = resources.createBuffer('invalid-count', 4);
  const insufficientPaddingBuffer = resources.createBuffer('insufficient-padding', 4);
  const weightSumBuffer = resources.createBuffer('weight-sum', 4);
  const parameterBuffer = resources.createParameterBuffer('parameters', 'float32', 9);
  let current: DensityGraph | null = null;
  let latestDesiredKey = '';
  let generation = 0;
  let destroyed = false;
  let buildFailed = false;
  let outputVisible = false;
  let displayExtent: readonly [number, number] = EMPTY_EXTENT;
  let currentWeights = new Float32Array(pointCount);
  let dirty = true;
  let summaryDirty = true;
  let parametersDirty = true;
  const swap = resources.track(new GraphSwap());

  const createSummary = (
    summaryGeneration: number,
    summaryOptions: NatureKernelDensityOptions,
    includedWeightCount: number
  ) =>
    new SummaryReader(
      resources,
      `density-${summaryGeneration}`,
      [
        {buffer: densityBuffer, size: CELL_COUNT * 4},
        {buffer: bandwidthBuffer, size: 4},
        {buffer: invalidCountBuffer, size: 4},
        {buffer: insufficientPaddingBuffer, size: 4},
        {buffer: weightSumBuffer, size: 4}
      ],
      bytes => {
        if (summaryGeneration !== generation || destroyed) return;
        const values = new Float32Array(bytes, 0, CELL_COUNT);
        let integral = 0;
        let peak = 0;
        let valid = 0;
        for (const value of values) {
          if (Number.isFinite(value)) {
            integral += value * CELL_SIZE * CELL_SIZE;
            peak = Math.max(peak, value);
            valid++;
          }
        }
        const reportedBandwidth = new Float32Array(bytes, CELL_COUNT * 4, 1)[0];
        const invalidCount = new Uint32Array(bytes, CELL_COUNT * 4 + 4, 1)[0];
        const insufficientPadding = new Uint32Array(bytes, CELL_COUNT * 4 + 8, 1)[0];
        const weightSum = new Float32Array(bytes, CELL_COUNT * 4 + 12, 1)[0];
        const displayedPrecision = summaryOptions.normalization === 'probability' ? 6 : 3;
        ctx.setReadout('included', `${weightSum.toLocaleString()} total weight`);
        ctx.setReadout('integral', integral.toFixed(3));
        ctx.setReadout('peak', `${(peak * 1e6).toFixed(displayedPrecision)} / km²`);
        ctx.setReadout('validCells', `${valid.toLocaleString()} / ${CELL_COUNT.toLocaleString()}`);
        ctx.setReadout('cell16', formatCellValue(values, 16, 16, displayedPrecision));
        ctx.setReadout('cell32', formatCellValue(values, 32, 32, displayedPrecision));
        ctx.setReadout('cell48', formatCellValue(values, 48, 48, displayedPrecision));
        ctx.setReadout('invalidRows', invalidCount.toLocaleString());
        const methodBandwidth =
          summaryOptions.bandwidthRule === 'fixed' ? summaryOptions.bandwidth : reportedBandwidth;
        const resolvedMethod =
          summaryOptions.kernel !== 'gaussian'
            ? 'exact'
            : summaryOptions.method === 'auto'
              ? !Number.isFinite(methodBandwidth)
                ? 'unresolved'
                : methodBandwidth <= (32 * CELL_SIZE) / 5 ||
                    methodBandwidth > (WIDTH * CELL_SIZE) / 2.5
                  ? 'exact'
                  : 'binned'
              : summaryOptions.method;
        ctx.setReadout('resolvedMethod', resolvedMethod);
        ctx.setReadout(
          'paddingStatus',
          resolvedMethod === 'exact'
            ? 'not used'
            : resolvedMethod === 'unresolved'
              ? 'unresolved bandwidth'
              : resolvedMethod === 'binned' && insufficientPadding
                ? 'insufficient'
                : summaryOptions.method === 'auto'
                  ? 'auto fallback available'
                  : 'sufficient'
        );
        if (summaryOptions.bandwidthRule !== 'fixed') {
          ctx.setReadout(
            'bandwidth',
            Number.isFinite(reportedBandwidth)
              ? `${reportedBandwidth.toFixed(0)} m (${summaryOptions.bandwidthRule})`
              : 'undefined (insufficient spread)'
          );
        }
        const validOutput =
          valid === CELL_COUNT &&
          invalidCount === 0 &&
          resolvedMethod !== 'unresolved' &&
          (resolvedMethod !== 'binned' || insufficientPadding === 0) &&
          Number.isFinite(weightSum) &&
          weightSum > 0 &&
          includedWeightCount > 0;
        if (validOutput) {
          displayExtent = [0, peak * 1e6];
          ctx.setLegendExtent('density', displayExtent);
        }
        outputVisible = validOutput;
        if (!validOutput) {
          ctx.setReadout('integral', null);
          ctx.setReadout('peak', null);
          ctx.setReadout('cell16', null);
          ctx.setReadout('cell32', null);
          ctx.setReadout('cell48', null);
        }
        ctx.requestLayers();
      }
    );

  const clearOutput = () => {
    ctx.setReadout('integral', null);
    ctx.setReadout('peak', null);
    ctx.setReadout('validCells', null);
    ctx.setReadout('cell16', null);
    ctx.setReadout('cell32', null);
    ctx.setReadout('cell48', null);
    ctx.setReadout('invalidRows', null);
    ctx.setReadout('paddingStatus', null);
    ctx.setReadout('resolvedMethod', null);
    displayExtent = EMPTY_EXTENT;
    ctx.setLegendExtent('density', EMPTY_EXTENT);
    outputVisible = false;
    ctx.requestLayers();
  };

  const writeWeights = () => {
    const grade = ctx.options.weight === 'researchGrade';
    let count = 0;
    for (let row = 0; row < pointCount; row++) {
      const included = !grade || researchGrade[sourceIndices[row]] !== 0;
      currentWeights[row] = included ? 1 : 0;
      count += included ? 1 : 0;
    }
    weightBuffer.write(currentWeights);
    ctx.setReadout('included', count.toLocaleString());
    ctx.setReadout('zeroWeight', (pointCount - count).toLocaleString());
    ctx.setReadout('excluded', (positions.length / 2 - pointCount).toLocaleString());
  };

  const build = async (
    options: NatureKernelDensityOptions,
    summaryGeneration: number,
    summaryWeightCount: number
  ): Promise<DensityGraph> => {
    const buildOptions = {...options};
    const graph = new GPUCommandGraph<void>(ctx.device, {id: 'nature-kernel-density'});
    const densityView = importGraphBuffer(graph, 'density', densityBuffer, 'float32', CELL_COUNT);
    const positionsView = importGraphBuffer(
      graph,
      'positions',
      positionBuffer,
      'float32x2',
      pointCount
    );
    const weightsView = importGraphBuffer(graph, 'weights', weightBuffer, 'float32', pointCount);
    const bandwidthView = importGraphBuffer(graph, 'bandwidth', bandwidthBuffer, 'float32', 1);
    const invalidCountView = importGraphBuffer(
      graph,
      'invalid-count',
      invalidCountBuffer,
      'uint32',
      1
    );
    const insufficientPaddingView = importGraphBuffer(
      graph,
      'insufficient-padding',
      insufficientPaddingBuffer,
      'uint32',
      1
    );
    const parameters = parameterBuffer.importToGraph(graph);
    const method = buildOptions.kernel === 'gaussian' ? buildOptions.method : 'exact';
    graph.add(
      new GPUKernelDensity({
        id: 'nature-kernel-density',
        spatialContext: FRAME.spatialContext,
        positions: positionsView,
        weights: weightsView,
        parameters,
        grid: GRID,
        indexGridSize: INDEX_GRID_SIZE,
        indexBounds: EXTENT,
        kernel: buildOptions.kernel,
        edgeCorrection: buildOptions.edgeCorrection,
        normalization: buildOptions.normalization,
        method,
        binnedPadding: WIDTH,
        ...(buildOptions.bandwidthRule === 'fixed'
          ? {}
          : {bandwidthRule: buildOptions.bandwidthRule}),
        output: {
          values: densityView,
          weightSum: importGraphBuffer(graph, 'weight-sum', weightSumBuffer, 'float32', 1),
          ...(buildOptions.bandwidthRule === 'fixed' ? {} : {bandwidth: bandwidthView}),
          status: {
            invalidCount: invalidCountView,
            ...(method === 'exact' ? {} : {insufficientPadding: insufficientPaddingView})
          }
        }
      })
    );
    const compiled = await graph.compileAsync();
    resources.track(compiled);
    const summary = createSummary(summaryGeneration, buildOptions, summaryWeightCount);
    return {graph: compiled, density: densityBuffer, summary};
  };

  const optionsKey = (options: NatureKernelDensityOptions) =>
    `${options.kernel}:${options.edgeCorrection}:${options.normalization}:${options.method}:${options.bandwidthRule}`;
  writeWeights();
  parameterBuffer.write(
    getGPUKernelDensityParameterValues({bandwidth: ctx.options.bandwidth, window: EXTENT}, GRID)
  );
  parametersDirty = false;
  current = await build(
    {...ctx.options},
    generation,
    currentWeights.reduce((sum, value) => sum + (value > 0 ? 1 : 0), 0)
  );
  latestDesiredKey = optionsKey(ctx.options);
  ctx.setReadout('records', (positions.length / 2).toLocaleString());
  ctx.setReadout('windowRecords', pointCount.toLocaleString());
  ctx.setReadout('bandwidth', `${ctx.options.bandwidth} m`);
  ctx.setReadout('integral', 'waiting for GPU');

  const requestBuild = (options: NatureKernelDensityOptions) => {
    const snapshot = {...options};
    const nextKey = optionsKey(snapshot);
    latestDesiredKey = nextKey;
    generation++;
    buildFailed = false;
    const buildGeneration = generation;
    current?.summary.release();
    if (current) {
      current.summary = createSummary(
        buildGeneration,
        snapshot,
        currentWeights.reduce((sum, value) => sum + (value > 0 ? 1 : 0), 0)
      );
    }
    clearOutput();
    ctx.setReadout('integral', 'rebuilding');
    ctx.setReadout('bandwidth', null);
    return swap.request({
      build: () =>
        build(
          snapshot,
          buildGeneration,
          currentWeights.reduce((sum, value) => sum + (value > 0 ? 1 : 0), 0)
        ),
      commit: next => {
        if (buildGeneration !== generation || nextKey !== latestDesiredKey) {
          next.summary.release();
          resources.release(next.graph);
          return;
        }
        current?.summary.release();
        if (current) resources.release(current.graph);
        current = next;
        buildFailed = false;
        dirty = true;
        summaryDirty = true;
        ctx.requestLayers();
      },
      discard: next => {
        next.summary.release();
        resources.release(next.graph);
      }
    });
  };

  return {
    getCompiledGraphs: () => (current ? [current.graph] : []),
    encode(commandEncoder) {
      const options = ctx.options;
      if (parametersDirty) {
        parameterBuffer.write(
          getGPUKernelDensityParameterValues({bandwidth: options.bandwidth, window: EXTENT}, GRID)
        );
        parametersDirty = false;
      }
      if (dirty && current && !swap.pending && !buildFailed) {
        current.graph.encode(commandEncoder, {parameters: undefined});
        dirty = false;
      }
      if (current && !swap.pending && !buildFailed && summaryDirty && !dirty) {
        current.summary.request(commandEncoder);
        summaryDirty = false;
      } else {
        current?.summary.flush(commandEncoder);
      }
      // Keep the displayed scientific setting current during rule or slider changes.
      if (options.bandwidthRule === 'fixed') ctx.setReadout('bandwidth', `${options.bandwidth} m`);
    },
    getLayers(): Layer[] {
      if (!current || swap.pending) return [];
      const dark = ctx.ground() === 'dark';
      const ramp = ctx.options.ramp;
      const layers: Layer[] = [];
      if (outputVisible && ctx.options.showSurface) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'nature-kernel-density-raster',
            analysisFrame: ANALYSIS_FRAME,
            gridSize: [WIDTH, HEIGHT],
            bounds: EXTENT,
            rowOrigin: 'north',
            values: densityBuffer,
            valueFormat: 'float32',
            colormap: ramp,
            valueScale: 1e6,
            valueRange: displayExtent,
            rampRange: [0.15, 1],
            color: dark ? [255, 255, 255, 230] : [255, 255, 255, 245]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'nature-kernel-density-observations',
          analysisFrame: ANALYSIS_FRAME,
          positions: positionBuffer,
          instanceCount: pointCount,
          radiusPixels: 1.4,
          color: [191, 212, 220, 115]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'nature-kernel-density-window',
          analysisFrame: ANALYSIS_FRAME,
          segments: windowSegments,
          instanceCount: 4,
          widthPixels: 1.3,
          color: dark ? [245, 240, 226, 220] : [30, 38, 44, 200],
          dashArray: [5, 4]
        })
      );
      return layers;
    },
    setOption(id, _value, state) {
      if (id === 'bandwidth') {
        generation++;
        current?.summary.release();
        if (current)
          current.summary = createSummary(
            generation,
            {...state},
            currentWeights.reduce((sum, value) => sum + (value > 0 ? 1 : 0), 0)
          );
        clearOutput();
        dirty = true;
        summaryDirty = true;
        if (swap.pending) {
          const rebuild = requestBuild({...state});
          const rebuildGeneration = generation;
          void rebuild.catch(error => {
            if (rebuildGeneration !== generation) return;
            buildFailed = true;
            clearOutput();
            ctx.setReadout('bandwidth', null);
            ctx.setStatus(`Kernel density graph rebuild failed: ${String(error)}`);
          });
        }
        parametersDirty = true;
      } else if (id === 'weight') {
        writeWeights();
        generation++;
        current?.summary.release();
        if (current)
          current.summary = createSummary(
            generation,
            {...state},
            currentWeights.reduce((sum, value) => sum + (value > 0 ? 1 : 0), 0)
          );
        clearOutput();
        dirty = true;
        summaryDirty = true;
        if (swap.pending) {
          const rebuild = requestBuild({...state});
          const rebuildGeneration = generation;
          void rebuild.catch(error => {
            if (rebuildGeneration !== generation) return;
            buildFailed = true;
            clearOutput();
            ctx.setReadout('bandwidth', null);
            ctx.setStatus(`Kernel density graph rebuild failed: ${String(error)}`);
          });
        }
      } else if (id === 'ramp' || id === 'showSurface') {
        ctx.requestLayers();
      } else {
        let options = {...state};
        if (id === 'kernel' && options.kernel !== 'gaussian' && options.method !== 'exact') {
          options = {...options, method: 'exact'};
          ctx.setOptions({method: 'exact'}, {notify: true});
        }
        const rebuild = requestBuild(options);
        const rebuildGeneration = generation;
        if (rebuild) {
          void rebuild.catch(error => {
            if (rebuildGeneration !== generation) return;
            buildFailed = true;
            clearOutput();
            ctx.setReadout('bandwidth', null);
            ctx.setStatus(`Kernel density graph rebuild failed: ${String(error)}`);
          });
        }
      }
    },
    destroy() {
      destroyed = true;
      current?.summary.release();
      resources.destroy();
    }
  };
}

function formatCellValue(
  values: Float32Array,
  column: number,
  row: number,
  decimalPlaces: number
): string {
  const value = values[row * WIDTH + column];
  return Number.isFinite(value)
    ? `${(value * 1e6).toFixed(decimalPlaces)} / km²`
    : 'outside window';
}
