// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {CommandEncoder, Buffer} from '@luma.gl/core';
import {GPUDorlingCartogram} from '@luma-contributors/graph/gpu-spatial-analysis';
import {GPUCommandGraph, type GraphDataView} from '@luma-contributors/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {GraphSwap} from '../../engine/graph-swap';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {getCountyDorlingAuthoredVertexMeanAnchor} from './county-dorling-anchor';
import type {SceneContext, SceneInstance} from '../scene';

export type CountyDorlingOptions = {
  display: 'anchors' | 'population-at-origins' | 'displaced';
  attraction: '0' | '0.01' | '0.05';
  iterationBudget: '0' | '25' | '100' | '256';
};

const ROW_COUNT = 21;
const CARTOGRAM_RADIUS_METERS = 20_000;
const MAXIMUM_POPULATION = 1_000_000;
const AREA_SCALE = (Math.PI * CARTOGRAM_RADIUS_METERS ** 2) / MAXIMUM_POPULATION;
const OVERLAP_TOLERANCE_METERS = 100;

type CountyDorlingVariant = {
  resources: SpatialAnalysisResources;
  compiled: Awaited<ReturnType<GPUCommandGraph<void>['compileAsync']>>;
  reader: SummaryReader;
  anchors: Buffer;
  displaced: Buffer;
  values: Buffer;
  radii: Buffer;
  leaderLines: Buffer;
  statusGeneration: number;
};

/** GPU-relaxed Dorling circles retain source IDs and report the actual bounded run status. */
export async function createCountyDorling(
  ctx: SceneContext<CountyDorlingOptions>
): Promise<SceneInstance<CountyDorlingOptions>> {
  const dataset = ctx.datasets.get('ncovr-new-jersey');
  const frame = {kind: 'azimuthal-equidistant' as const, origin: [-74.55, 40.05] as const};
  const projection = createAnalysisFrameProjection(frame);
  const datasetResources = new SpatialAnalysisResources(ctx.device, 'county-dorling-input');
  const geojson = dataset.geojson;
  if (!geojson || geojson.features.length !== ROW_COUNT)
    throw new Error('Expected the complete 21-unit NCOVR historical geography');
  const names = geojson.features.map(feature =>
    String(feature.properties?.NAME ?? feature.properties?.name ?? 'County unit')
  );
  const initialPositions = new Float32Array(ROW_COUNT * 2);
  const populations = dataset.column<Float32Array>('population1990');
  const sourceIds = dataset.column<Uint32Array>('fips');
  for (let row = 0; row < ROW_COUNT; row++) {
    const [longitude, latitude] = getCountyDorlingAuthoredVertexMeanAnchor(geojson.features[row]!);
    initialPositions.set(projection.project(longitude, latitude), row * 2);
  }
  const anchorBuffer = datasetResources.createBuffer(
    'geographic-anchor-positions',
    initialPositions
  );
  const valueBuffer = datasetResources.createBuffer('1990-population-counts', populations);
  const sourceIdBuffer = datasetResources.createBuffer('NCOVR-source-FIPS', sourceIds);
  let variantNumber = 0;
  let generation = 0;

  const buildVariant = async (
    attraction: number,
    iterationBudget: number,
    variantGeneration: number
  ): Promise<CountyDorlingVariant> => {
    const resources = new SpatialAnalysisResources(
      ctx.device,
      `county-dorling-run-${++variantNumber}`
    );
    const displaced = resources.createBuffer('displaced-centers', ROW_COUNT * 2 * 4);
    const radii = resources.createBuffer('population-radii', ROW_COUNT * 4);
    const copiedIds = resources.createBuffer('cartogram-source-FIPS', ROW_COUNT * 4);
    const metrics = resources.createBuffer('cartogram-metrics', 2 * 4);
    const status = resources.createBuffer('cartogram-status', 4 * 4);
    const leaderLines = resources.createBuffer('anchor-leader-lines', ROW_COUNT * 4 * 4);
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `ncovr-new-jersey-dorling-${variantNumber}`
    });
    const importedViews = new Map<Buffer, GraphDataView>();
    const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => {
      const existing = importedViews.get(buffer);
      if (existing) return existing as GraphDataView<Format>;
      const imported = importGraphBuffer(graph, name, buffer, format, length);
      importedViews.set(buffer, imported);
      return imported;
    };
    const anchorView = view('anchors', anchorBuffer, 'float32x2', ROW_COUNT);
    const displacedView = view('displaced', displaced, 'float32x2', ROW_COUNT);
    const valueView = view('population', valueBuffer, 'float32', ROW_COUNT);
    const sourceIdView = view('source-ids', sourceIdBuffer, 'uint32', ROW_COUNT);
    const outputMetrics = view('metrics', metrics, 'float32x2', 1);
    const outputStatus = view('status', status, 'uint32', 4);
    graph.add(
      new GPUDorlingCartogram({
        id: `ncovr-dorling-${variantNumber}`,
        spatialContext: projection.spatialContext,
        positions: anchorView,
        values: valueView,
        sourceIds: sourceIdView,
        areaScale: AREA_SCALE,
        maximumIterations: iterationBudget,
        overlapTolerance: OVERLAP_TOLERANCE_METERS,
        attraction,
        output: {
          positions: displacedView,
          radii: view('radii', radii, 'float32', ROW_COUNT),
          sourceIds: view('output-ids', copiedIds, 'uint32', ROW_COUNT),
          metrics: outputMetrics,
          status: outputStatus
        }
      })
    );
    const leaderView = view('leader-lines', leaderLines, 'float32', ROW_COUNT * 4);
    addKernelPass(graph, {
      id: `write-geographic-anchor-leaders-${variantNumber}`,
      invocationCount: ROW_COUNT,
      bindings: [
        {name: 'anchors', view: anchorView, type: 'f32', access: 'read'},
        {name: 'centers', view: displacedView, type: 'f32', access: 'read'},
        {name: 'segments', view: leaderView, type: 'f32', access: 'read_write'}
      ],
      body: 'let row = index * 4u; segments[segmentsOffset + row] = anchors[anchorsOffset + index * 2u]; segments[segmentsOffset + row + 1u] = anchors[anchorsOffset + index * 2u + 1u]; segments[segmentsOffset + row + 2u] = centers[centersOffset + index * 2u]; segments[segmentsOffset + row + 3u] = centers[centersOffset + index * 2u + 1u];'
    });
    let compiled: Awaited<ReturnType<GPUCommandGraph<void>['compileAsync']>>;
    try {
      compiled = resources.track(await graph.compileAsync());
    } catch (error) {
      resources.destroy();
      throw error;
    }
    const reader = new SummaryReader(
      resources,
      `dorling-status-${variantNumber}`,
      [
        {buffer: metrics, size: 2 * 4},
        {buffer: status, size: 4 * 4},
        {buffer: displaced, size: ROW_COUNT * 2 * 4},
        {buffer: radii, size: ROW_COUNT * 4},
        {buffer: copiedIds, size: ROW_COUNT * 4}
      ],
      bytes => {
        if (variantGeneration !== generation) return;
        const metricValues = new Float32Array(bytes, 0, 2);
        const statusValues = new Uint32Array(bytes, 8, 4);
        const returnedCenters = new Float32Array(bytes, 24, ROW_COUNT * 2);
        const returnedRadii = new Float32Array(bytes, 24 + ROW_COUNT * 8, ROW_COUNT);
        const returnedIds = new Uint32Array(bytes, 24 + ROW_COUNT * 12, ROW_COUNT);
        displayedCenters = returnedCenters.slice();
        displayedRadii = returnedRadii.slice();
        displayedIds = returnedIds.slice();
        let orderedIds = 0;
        let maximumRadiusRelativeError = 0;
        let radiiValid = true;
        for (let row = 0; row < ROW_COUNT; row++) {
          if (returnedIds[row] === sourceIds[row]) orderedIds++;
          const sourceRow = sourceIds.indexOf(returnedIds[row]!);
          const actualPopulation = sourceRow < 0 ? Number.NaN : populations[sourceRow]!;
          const expectedRadius = Math.sqrt((actualPopulation * AREA_SCALE) / Math.PI);
          const actualRadius = returnedRadii[row]!;
          const expectedArea = expectedRadius * expectedRadius;
          const actualArea = actualRadius * actualRadius;
          const relativeError = Math.abs(actualArea - expectedArea) / expectedArea;
          if (
            !Number.isFinite(relativeError) ||
            !Number.isFinite(actualRadius) ||
            actualRadius <= 0
          ) {
            radiiValid = false;
          } else {
            maximumRadiusRelativeError = Math.max(maximumRadiusRelativeError, relativeError);
          }
        }
        const idsValid = orderedIds === ROW_COUNT;
        radiiValid = radiiValid && maximumRadiusRelativeError <= 1e-5;
        returnedOutputValid = idsValid && radiiValid;
        if (!returnedOutputValid) {
          displayedCenters = null;
          displayedRadii = null;
          displayedIds = null;
        }
        const exhausted = statusValues[2] !== 0;
        const converged = statusValues[1] !== 0;
        const statusText =
          iterationBudget === 0
            ? `0 iterations; starting overlap; ${statusValues[3]} excluded`
            : `${statusValues[0]} / ${iterationBudget} iterations; ${converged ? 'overlap tolerance met' : exhausted ? 'iteration budget exhausted' : 'stopped before tolerance'}; ${statusValues[3]} excluded`;
        ctx.setReadout('maximumOverlap', `${metricValues[0]!.toFixed(0)} m`);
        ctx.setReadout('maximumDisplacement', `${metricValues[1]!.toFixed(0)} m`);
        ctx.setReadout('solverStatus', statusText);
        ctx.setReadout(
          'circleCount',
          `${ROW_COUNT - statusValues[3]} / ${ROW_COUNT} eligible; ordered source IDs ${orderedIds} / ${ROW_COUNT} ${idsValid ? 'valid' : 'INVALID'}; circle area check max relative error ${maximumRadiusRelativeError.toExponential(2)} ${radiiValid ? 'valid' : 'INVALID'}`
        );
        ctx.requestLayers();
        ctx.refreshTooltip();
      }
    );
    return {
      resources,
      compiled,
      reader,
      anchors: anchorBuffer,
      displaced,
      values: valueBuffer,
      radii,
      leaderLines,
      statusGeneration: variantGeneration
    };
  };

  let active = await buildVariant(
    Number(ctx.options.attraction),
    Number(ctx.options.iterationBudget),
    ++generation
  );
  const graphSwap = datasetResources.track(new GraphSwap());
  let dirty = true;
  let displayedCenters: Float32Array | null = null;
  let displayedRadii: Float32Array | null = null;
  let displayedIds: Uint32Array | null = null;
  let returnedOutputValid = false;
  let legendScale = Number.NaN;
  const updateLegend = () => {
    const metersPerPixel = ctx.getMetersPerPixel();
    if (Math.abs(metersPerPixel - legendScale) < metersPerPixel * 0.01) return;
    legendScale = metersPerPixel;
    const maximumRadiusPixels = CARTOGRAM_RADIUS_METERS / metersPerPixel;
    const entries = [1_000_000, 500_000, 250_000].map(value => ({
      radiusPixels:
        (CARTOGRAM_RADIUS_METERS * Math.sqrt(value / MAXIMUM_POPULATION)) / metersPerPixel,
      label: `${Math.round(value / 1_000)}k`
    }));
    ctx.setLegendData('countyDorling', {entries, maximumRadiusPixels});
  };
  updateLegend();

  return {
    getCompiledGraphs: () => [active.compiled],
    encode(commandEncoder: CommandEncoder) {
      updateLegend();
      if (dirty) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        active.reader.request(commandEncoder);
      } else active.reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      updateLegend();
      const display = ctx.options.display;
      const layers: Layer[] = [];
      if (
        display === 'displaced' &&
        returnedOutputValid &&
        displayedCenters &&
        displayedRadii &&
        displayedIds
      ) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'county-dorling-geographic-leaders',
            analysisFrame: frame,
            segments: active.leaderLines,
            instanceCount: ROW_COUNT,
            color: [54, 68, 79, 125],
            widthPixels: 1
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'county-dorling-geographic-anchors',
          analysisFrame: frame,
          positions: active.anchors,
          instanceCount: ROW_COUNT,
          shape: 'ring',
          radiusPixels: 4.5,
          color: [36, 48, 57, 245],
          outlineWidthPixels: 1.5
        })
      );
      if (
        display !== 'anchors' &&
        returnedOutputValid &&
        displayedCenters &&
        displayedRadii &&
        displayedIds
      ) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'county-dorling-population-circles',
            analysisFrame: frame,
            positions: display === 'displaced' ? active.displaced : active.anchors,
            instanceCount: ROW_COUNT,
            radiusMeters: CARTOGRAM_RADIUS_METERS,
            sizeValues: display === 'displaced' ? active.radii : valueBuffer,
            sizeMaximumValue:
              display === 'displaced' ? CARTOGRAM_RADIUS_METERS : MAXIMUM_POPULATION,
            sizeScale: display === 'displaced' ? 'linear' : 'sqrt',
            fillOpacity: 0.72,
            color: [0, 114, 178, 215],
            outlineColor: [17, 54, 73, 235],
            outlineWidthPixels: 1
          })
        );
      }
      return layers;
    },
    async setOption(id, _value, state) {
      if (id === 'display') {
        ctx.requestLayers();
        return;
      }
      if (id !== 'attraction' && id !== 'iterationBudget') return;
      displayedCenters = null;
      displayedRadii = null;
      displayedIds = null;
      returnedOutputValid = false;
      ctx.setReadout('maximumOverlap', null);
      ctx.setReadout('maximumDisplacement', null);
      ctx.setReadout('solverStatus', 'Compiling requested run');
      ctx.setReadout('circleCount', null);
      dirty = true;
      const nextGeneration = ++generation;
      try {
        await graphSwap.request({
          build: async () => {
            return await buildVariant(
              Number(state.attraction),
              Number(state.iterationBudget),
              nextGeneration
            );
          },
          commit: next => {
            const previous = active;
            active = next;
            dirty = true;
            displayedCenters = null;
            displayedRadii = null;
            displayedIds = null;
            ctx.requestLayers();
            previous.reader.release();
            previous.resources.destroy();
          },
          discard: next => {
            next.reader.release();
            next.resources.destroy();
          }
        });
      } catch {
        if (generation === nextGeneration) {
          generation = active.statusGeneration;
          ctx.setReadout('solverStatus', 'Compile failed; previous GPU run retained');
          dirty = true;
          ctx.requestLayers();
        }
      }
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const display = ctx.options.display;
      let nearestRow = -1;
      let nearestDistance = Infinity;
      let hitDisplayedCircle = false;
      if (display !== 'anchors' && displayedCenters && displayedRadii && displayedIds) {
        for (let row = 0; row < displayedIds.length; row++) {
          const sourceRow = sourceIds.indexOf(displayedIds[row]!);
          if (sourceRow < 0) continue;
          const centerX =
            display === 'displaced' ? displayedCenters[row * 2]! : initialPositions[sourceRow * 2]!;
          const centerY =
            display === 'displaced'
              ? displayedCenters[row * 2 + 1]!
              : initialPositions[sourceRow * 2 + 1]!;
          const distance = Math.hypot(centerX - x, centerY - y);
          const radius = displayedRadii[row]!;
          if (distance <= radius && distance < nearestDistance) {
            nearestRow = sourceRow;
            nearestDistance = distance;
            hitDisplayedCircle = true;
          }
        }
      }
      if (!hitDisplayedCircle || nearestRow < 0) {
        if (display !== 'anchors') return null;
        for (let row = 0; row < ROW_COUNT; row++) {
          const distance = Math.hypot(
            initialPositions[row * 2]! - x,
            initialPositions[row * 2 + 1]! - y
          );
          if (distance < nearestDistance) {
            nearestRow = row;
            nearestDistance = distance;
          }
        }
      }
      if (nearestRow < 0) return null;
      const displacedRow = displayedIds?.indexOf(sourceIds[nearestRow]!) ?? -1;
      const displacement =
        displacedRow >= 0 && displayedCenters
          ? Math.hypot(
              displayedCenters[displacedRow * 2]! - initialPositions[nearestRow * 2]!,
              displayedCenters[displacedRow * 2 + 1]! - initialPositions[nearestRow * 2 + 1]!
            )
          : null;
      const match =
        display === 'anchors'
          ? 'Nearest geographic anchor (not a circle hit)'
          : `Hit displayed ${display === 'displaced' ? 'displaced' : 'origin'} population circle`;
      return `${match}: ${names[nearestRow]} · exact source FIPS ${sourceIds[nearestRow]}; 1990 population ${Math.round(populations[nearestRow]!).toLocaleString()} people; GPU center displacement ${displacement === null ? 'pending' : `${displacement.toFixed(0)} m`}. Circle area follows πr² = population × ${AREA_SCALE.toPrecision(5)} m² per person; anchor is the authored polygon vertex mean, not guaranteed interior.`;
    },
    destroy() {
      active.reader.release();
      active.resources.destroy();
      datasetResources.destroy();
    }
  };
}
