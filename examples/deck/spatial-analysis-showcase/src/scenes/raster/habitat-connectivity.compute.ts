// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUElementwise,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {
  GPURasterConnectedComponents,
  GPURasterDenseComponents,
  GPURasterErosion
} from '@luma.gl/graph/gpu-raster';
import {
  GPURasterPatchMetrics,
  GPURasterWalkingCost,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {SpatialAnalysisResources} from '../../engine/resources';
import {GraphSwap} from '../../engine/graph-swap';
import {addGraphCopy} from '../../engine/graph-buffers';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, createUtmFrame, loadClassRaster, UtmRasterLayer} from './b16-common';
import {HABITAT_PATCH_COLORS, WORLD_COVER_CLASSES} from './habitat-connectivity-style';

export type HabitatConnectivityOptions = {
  layer: 'cover' | 'mask' | 'core' | 'patches' | 'cost';
  habitat: 'tree' | 'woody' | 'broad';
  countRasterBorder: boolean;
  coreCells: number;
  resistance: 'mild' | 'strong';
  opacity: number;
};

const GRID_SIZE = 64;
const PATCH_CAPACITY = 128;
const CORE_RADII = [1, 2, 3, 4] as const;
const MILD_RESISTANCE: Record<number, number> = {
  10: 1,
  20: 1.5,
  30: 2,
  40: 3,
  50: 8,
  60: 4,
  70: 5,
  80: 20,
  90: 4,
  95: 2,
  100: 3
};
const STRONG_RESISTANCE: Record<number, number> = {
  10: 1,
  20: 2,
  30: 6,
  40: 15,
  50: 80,
  60: 10,
  70: 20,
  80: 120,
  90: 15,
  95: 3,
  100: 4
};

type HabitatGraphBuffers = {
  mask: Buffer;
  cover: Buffer;
  sparse: Buffer;
  sparseValidity: Buffer;
  converged: Buffer;
  labels: Buffer;
  labelValidity: Buffer;
  componentCount: Buffer;
  overflow: Buffer;
  pixelCounts: Buffer;
  areas: Buffer;
  perimeters: Buffer;
  costs: Buffer;
  costValidity: Buffer;
  costConverged: Buffer;
  revision: Buffer;
  analysisLabels: Buffer;
  patchClasses: Buffer;
  proximityCore: Buffer;
  proximityMetrics: Buffer;
  landscape: Buffer;
  landscapeCohesion: Buffer;
  coreVariants: Buffer[];
  coreValidities: Buffer[];
  friction: Buffer;
  elevation: Buffer;
};

/**
 * Habitat-connectivity scenarios from a bounded 20 m WorldCover crop. Patch topology and shape
 * are GPU products. A separate GPU walking-cost surface uses explicit class resistance assumptions
 * and the DEM; it is a scenario surface, not a measurement of animal movement or habitat quality.
 */
async function createHabitatConnectivityInstance(
  ctx: SceneContext<HabitatConnectivityOptions>,
  isCurrent: () => boolean
): Promise<SceneInstance<HabitatConnectivityOptions>> {
  const dataset = ctx.datasets.get('dixie-fire');
  ctx.setStatus('Loading Dixie Fire WorldCover');
  const worldCover = await loadClassRaster(dataset.info.id, 'worldcover.png', ctx.signal);
  ctx.signal.throwIfAborted();
  const manifest = dataset.manifest as unknown as {
    rasters: {
      dem: {
        width: number;
        height: number;
        boundsProjected: [number, number, number, number];
        cellSizeM: number;
      };
      worldcover: {
        width: number;
        height: number;
        boundsProjected: [number, number, number, number];
        cellSizeM: number;
      };
    };
  };
  const dem = dataset.raster?.values;
  if (!(dem instanceof Float32Array)) throw new Error('dixie-fire needs its Terrarium DEM');
  const sourceFrame = createUtmFrame(dataset.defaultOrigin, manifest.rasters.dem);
  const originalCenter = sourceFrame.lngLatToCell(-120.9511, 40.1396);
  const startColumn = Math.max(
    0,
    Math.min(
      manifest.rasters.worldcover.width - GRID_SIZE,
      Math.floor(originalCenter[0] - GRID_SIZE / 2)
    )
  );
  const startRow = Math.max(
    0,
    Math.min(
      manifest.rasters.worldcover.height - GRID_SIZE,
      Math.floor(originalCenter[1] - GRID_SIZE / 2)
    )
  );
  const cellCount = GRID_SIZE * GRID_SIZE;
  const classes = new Uint8Array(cellCount);
  const coverValues = new Uint32Array(cellCount);
  const elevation = new Float32Array(cellCount);
  const bounds = manifest.rasters.dem;
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let column = 0; column < GRID_SIZE; column++) {
      const sourceIndex =
        (startRow + row) * manifest.rasters.worldcover.width + startColumn + column;
      const demIndex = (startRow + row) * manifest.rasters.dem.width + startColumn + column;
      const index = row * GRID_SIZE + column;
      classes[index] = worldCover.classes[sourceIndex];
      coverValues[index] = Math.max(
        0,
        WORLD_COVER_CLASSES.findIndex(({code}) => code === classes[index])
      );
      elevation[index] = dem[demIndex];
    }
  }
  const cellSize = bounds.cellSizeM;
  // Build the local display frame from this exact projected crop.
  const cropBounds: [number, number, number, number] = [
    manifest.rasters.worldcover.boundsProjected[0] + startColumn * cellSize,
    manifest.rasters.worldcover.boundsProjected[3] - (startRow + GRID_SIZE) * cellSize,
    manifest.rasters.worldcover.boundsProjected[0] + (startColumn + GRID_SIZE) * cellSize,
    manifest.rasters.worldcover.boundsProjected[3] - startRow * cellSize
  ];
  const frame = createUtmFrame(dataset.defaultOrigin, {
    boundsProjected: cropBounds,
    cellSizeM: cellSize,
    width: GRID_SIZE,
    height: GRID_SIZE
  });

  const rasterGrid = makeGPURasterMetadata({
    width: GRID_SIZE,
    height: GRID_SIZE,
    cellSize,
    origin: [-frame.halfWidth, frame.halfHeight]
  });
  const resources = new SpatialAnalysisResources(ctx.device, 'habitat');
  const buffers: HabitatGraphBuffers = {
    mask: resources.createBuffer('mask', new Uint32Array(cellCount)),
    cover: resources.createBuffer('cover-classes', new Uint32Array(cellCount)),
    sparse: resources.createBuffer('sparse-labels', cellCount * 4),
    sparseValidity: resources.createBuffer('sparse-validity', cellCount * 4),
    converged: resources.createBuffer('converged', 4),
    labels: resources.createBuffer('labels', cellCount * 4),
    labelValidity: resources.createBuffer('label-validity', cellCount * 4),
    componentCount: resources.createBuffer('component-count', 4),
    overflow: resources.createBuffer('overflow', 4),
    pixelCounts: resources.createBuffer('patch-pixels', PATCH_CAPACITY * 4),
    areas: resources.createBuffer('patch-areas', PATCH_CAPACITY * 4),
    perimeters: resources.createBuffer('patch-perimeters', PATCH_CAPACITY * 4),
    costs: resources.createBuffer('walking-costs', cellCount * 4),
    costValidity: resources.createBuffer('cost-validity', cellCount * 4),
    costConverged: resources.createBuffer('cost-converged', 4),
    revision: resources.createBuffer('summary-revision', new Uint32Array(1)),
    analysisLabels: resources.createBuffer('analysis-labels', cellCount * 4),
    patchClasses: resources.createBuffer('patch-classes', new Uint32Array(PATCH_CAPACITY)),
    proximityCore: resources.createBuffer('proximity-core', cellCount * 4),
    proximityMetrics: resources.createBuffer('proximity-metrics', PATCH_CAPACITY * 16),
    landscape: resources.createBuffer('landscape-summary', 16),
    landscapeCohesion: resources.createBuffer('landscape-cohesion', 4),
    coreVariants: CORE_RADII.map(radius => resources.createBuffer(`core-${radius}`, cellCount * 4)),
    coreValidities: CORE_RADII.map(radius =>
      resources.createBuffer(`core-${radius}-validity`, cellCount * 4)
    ),
    friction: resources.createBuffer('friction', new Float32Array(cellCount)),
    elevation: resources.createBuffer('elevation', elevation)
  };
  buffers.cover.write(coverValues);
  buffers.patchClasses.write(new Uint32Array(PATCH_CAPACITY));

  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'habitat-connectivity'});
  const view = createGraphViewer(graph);
  const word = (name: string, buffer: Buffer, length = cellCount) =>
    view(name, buffer, 'uint32', length);
  const float = (name: string, buffer: Buffer, length = cellCount) =>
    view(name, buffer, 'float32', length);
  const maskView = word('habitat-mask', buffers.mask);
  const maskBand = {
    id: 'habitat-mask',
    format: 'uint32' as const,
    storage: {kind: 'buffer' as const, values: maskView}
  };
  CORE_RADII.forEach((radius, index) => {
    new GPURasterErosion({
      id: `mask-core-${radius}`,
      width: GRID_SIZE,
      height: GRID_SIZE,
      input: maskBand,
      output: word(`core-${radius}`, buffers.coreVariants[index]),
      outputValidity: word(`core-${radius}-validity`, buffers.coreValidities[index]),
      mode: 'binary',
      radius,
      structuringElement: 'square',
      borderValue: 0,
      borderMode: 'constant'
    }).addToGraph(graph);
  });
  const sparseView = word('sparse-labels', buffers.sparse);
  const sparseValidityView = word('sparse-validity', buffers.sparseValidity);
  const convergedView = word('component-converged', buffers.converged, 1);
  const labelsView = word('dense-labels', buffers.labels);
  const labelValidityView = word('dense-validity', buffers.labelValidity);
  const componentCountView = word('component-count', buffers.componentCount, 1);
  const overflowView = word('overflow', buffers.overflow, 1);
  const connectivity = new GPURasterConnectedComponents({
    id: 'habitat-patches',
    width: GRID_SIZE,
    height: GRID_SIZE,
    input: {id: 'habitat-mask', format: 'uint32', storage: {kind: 'buffer', values: maskView}},
    output: sparseView,
    outputValidity: sparseValidityView,
    converged: convergedView,
    connectivity: 8
  });
  connectivity.addToGraph(graph);
  new GPURasterDenseComponents({
    id: 'dense-patches',
    width: GRID_SIZE,
    height: GRID_SIZE,
    input: sparseView,
    inputValidity: sparseValidityView,
    converged: convergedView,
    output: labelsView,
    outputValidity: labelValidityView,
    componentCount: componentCountView,
    overflow: overflowView,
    capacity: PATCH_CAPACITY
  }).addToGraph(graph);
  const onesBuffer = resources.createBuffer(
    'patch-label-offset',
    new Uint32Array(cellCount).fill(1)
  );
  const analysisLabelsView = word('analysis-labels', buffers.analysisLabels);
  graph.add(
    new GPUElementwise({
      id: 'zero-base-patch-labels',
      input: labelsView,
      inputB: word('patch-label-offset', onesBuffer),
      output: analysisLabelsView,
      operation: 'subtract'
    })
  );
  graph.add(
    new GPURasterPatchMetrics({
      id: 'patch-shape',
      grid: rasterGrid,
      countRasterBorder: ctx.options.countRasterBorder,
      labels: labelsView,
      validity: labelValidityView,
      status: {
        converged: convergedView,
        componentCount: componentCountView,
        overflow: overflowView
      },
      output: {
        pixelCounts: word('patch-pixels', buffers.pixelCounts, PATCH_CAPACITY),
        areas: float('patch-areas', buffers.areas, PATCH_CAPACITY),
        perimeters: float('patch-perimeters', buffers.perimeters, PATCH_CAPACITY)
      }
    })
  );
  const patchClassesView = word('patch-classes', buffers.patchClasses, PATCH_CAPACITY);
  const landscape = addLandscapeMetricsRecipe(graph, {
    id: 'landscape-metrics',
    patches: analysisLabelsView,
    patchClasses: patchClassesView,
    width: GRID_SIZE,
    height: GRID_SIZE,
    classCount: 1,
    cellSize: [cellSize, cellSize],
    connectivity: 8,
    countBoundary: true,
    countBackground: true
  });
  const proximity = addPatchProximityRecipe(graph, {
    id: 'patch-proximity',
    patches: analysisLabelsView,
    patchClasses: patchClassesView,
    width: GRID_SIZE,
    height: GRID_SIZE,
    cellSize: [cellSize, cellSize],
    edgeDepth: 1,
    searchRadius: 2000,
    distanceModel: 'cell-boundary',
    coreModel: 'raster-erosion',
    coreBoundary: 'edge'
  });
  addGraphCopy(
    graph,
    proximity.outputs.core,
    word('proximity-core-output', buffers.proximityCore),
    {id: 'publish-proximity-core', afterNode: 'patch-proximity-core-distance'}
  );
  addGraphCopy(
    graph,
    landscape.outputs.landscape,
    float('landscape-summary-output', buffers.landscape, 4),
    {id: 'publish-landscape-summary', afterNode: 'landscape-metrics-landscape-summary'}
  );
  addGraphCopy(
    graph,
    landscape.outputs.landscapeCohesion,
    float('landscape-cohesion-output', buffers.landscapeCohesion, 1),
    {
      id: 'publish-landscape-cohesion',
      afterNode: 'landscape-metrics-landscape-aggregation-cohesion'
    }
  );
  addGraphCopy(
    graph,
    proximity.outputs.metrics,
    float('proximity-metrics-output', buffers.proximityMetrics, PATCH_CAPACITY * 4),
    {id: 'publish-proximity-metrics', afterNode: 'patch-proximity-patch-summary'}
  );
  graph.add(
    new GPURasterWalkingCost({
      id: 'walking-cost-scenario',
      grid: rasterGrid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: float('elevation', buffers.elevation)}
      },
      friction: {
        id: 'friction',
        format: 'float32',
        storage: {kind: 'buffer', values: float('friction', buffers.friction)}
      },
      sources: proximity.outputs.core,
      model: {kind: 'grass', coefficients: [1, 0.05, 0.05, 0.05], downhillSlopeThreshold: -0.3},
      maximumIterations: 128,
      output: {
        costs: float('walking-costs', buffers.costs),
        validity: word('walking-cost-validity', buffers.costValidity),
        status: {converged: word('walking-cost-converged', buffers.costConverged, 1)}
      }
    })
  );
  const compiled = resources.track(await graph.compileAsync());
  if (ctx.signal.aborted) {
    resources.destroy();
    ctx.signal.throwIfAborted();
  }

  let dirty = true;
  let destroyed = false;
  let lastStateKey = '';
  let revision = 0;
  const summaryRevisionByteOffset =
    36 + cellCount * 4 * (2 + CORE_RADII.length) + PATCH_CAPACITY * 16;
  const summaryReader = new SummaryReader(
    resources,
    'habitat-summary',
    [
      {buffer: buffers.componentCount, size: 4},
      {buffer: buffers.overflow, size: 4},
      {buffer: buffers.converged, size: 4},
      {buffer: buffers.costConverged, size: 4},
      {buffer: buffers.landscape, size: 16},
      {buffer: buffers.landscapeCohesion, size: 4},
      {buffer: buffers.mask, size: cellCount * 4},
      ...buffers.coreVariants.map(buffer => ({buffer, size: cellCount * 4})),
      {buffer: buffers.proximityMetrics, size: PATCH_CAPACITY * 16},
      {buffer: buffers.costValidity, size: cellCount * 4},
      {buffer: buffers.revision, size: 4},
      {buffer: buffers.perimeters, size: PATCH_CAPACITY * 4}
    ],
    bytes => {
      if (destroyed || ctx.signal.aborted || !isCurrent()) return;
      const summary = new DataView(bytes);
      if (summary.getUint32(summaryRevisionByteOffset, true) !== revision) return;
      const patchCount = summary.getUint32(0, true);
      const patchStatus = summary.getUint32(4, true)
        ? ' · capacity exceeded'
        : summary.getUint32(8, true)
          ? ''
          : ' · not converged';
      ctx.setReadout('patchCount', `${patchCount.toLocaleString()}${patchStatus}`);
      if (summary.getUint32(4, true)) {
        ctx.setReadout('patchPerimeter', 'withheld · patch output capacity exceeded');
      } else if (!summary.getUint32(8, true)) {
        ctx.setReadout('patchPerimeter', 'withheld · connected-component labeling not converged');
      } else {
        const perimeters = new Float32Array(bytes, summaryRevisionByteOffset + 4, PATCH_CAPACITY);
        let totalPerimeter = 0;
        let validPerimeterCount = 0;
        for (let patch = 0; patch < patchCount; patch++) {
          const perimeter = perimeters[patch];
          if (!Number.isFinite(perimeter)) continue;
          totalPerimeter += perimeter;
          validPerimeterCount++;
        }
        const groupLabel =
          ctx.options.habitat === 'tree'
            ? 'tree cover + mangroves'
            : ctx.options.habitat === 'woody'
              ? 'tree + shrub + mangroves'
              : 'woody + grass + crops + wetland + moss/lichen';
        const cropRows = `${startRow}–${startRow + GRID_SIZE - 1}`;
        const cropColumns = `${startColumn}–${startColumn + GRID_SIZE - 1}`;
        const borderLabel = ctx.options.countRasterBorder
          ? 'including outer crop faces'
          : 'excluding only outer crop faces';
        ctx.setReadout(
          'patchPerimeter',
          validPerimeterCount === patchCount
            ? `${groupLabel} · ${Math.round(totalPerimeter).toLocaleString()} m total · ${patchCount.toLocaleString()} patches · ${borderLabel} · WorldCover source rows ${cropRows}, columns ${cropColumns}`
            : `withheld · ${validPerimeterCount} of ${patchCount} patch perimeter values finite`
        );
      }
      let offset = 16;
      const cohesion = new Float32Array(bytes, offset + 16, 1)[0];
      offset += 20;
      const selectedCoreRadius = ctx.options.coreCells;
      let interiorCells = 0;
      const mask = new Uint32Array(bytes, offset, cellCount);
      if (selectedCoreRadius === 0) {
        for (const value of mask) interiorCells += value;
      }
      offset += cellCount * 4;
      for (let radius = 1; radius <= CORE_RADII.length; radius++) {
        const core = new Uint32Array(bytes, offset, cellCount);
        if (radius === selectedCoreRadius) {
          for (const value of core) interiorCells += value;
        }
        offset += cellCount * 4;
      }
      ctx.setReadout('interiorCells', interiorCells.toLocaleString());
      const proximityMetrics = new Float32Array(bytes, offset, PATCH_CAPACITY * 4);
      let nearestSeparation = Number.POSITIVE_INFINITY;
      let sourceCells = 0;
      for (let patch = 0; patch < patchCount; patch++) {
        sourceCells += proximityMetrics[patch * 4 + 1] / (cellSize * cellSize);
        const separation = proximityMetrics[patch * 4 + 2];
        if (Number.isFinite(separation))
          nearestSeparation = Math.min(nearestSeparation, separation);
      }
      ctx.setReadout(
        'patchSeparation',
        Number.isFinite(nearestSeparation)
          ? `${Math.round(nearestSeparation).toLocaleString()} m`
          : 'no other patch within search radius'
      );
      const costValidityOffset = offset + PATCH_CAPACITY * 16;
      let validCostCells = 0;
      for (const value of new Uint32Array(bytes, costValidityOffset, cellCount)) {
        validCostCells += value;
      }
      if (!sourceCells) {
        ctx.setReadout('costStatus', 'no source cells; cost is not interpretable');
      } else if (summary.getUint32(12, true)) {
        ctx.setReadout(
          'costStatus',
          `converged · ${validCostCells.toLocaleString()} valid cells from ${Math.round(sourceCells).toLocaleString()} sources`
        );
      } else {
        ctx.setReadout('costStatus', 'not converged; cost cells withheld');
      }
      ctx.setReadout(
        'landscapeCohesion',
        Number.isFinite(cohesion) ? `${cohesion.toFixed(1)}%` : 'undefined for this mask'
      );
    }
  );
  const coverSelected = (code: number, habitat: HabitatConnectivityOptions['habitat']): boolean =>
    habitat === 'tree'
      ? code === 10 || code === 95
      : habitat === 'woody'
        ? code === 10 || code === 20 || code === 95
        : code === 10 ||
          code === 20 ||
          code === 30 ||
          code === 40 ||
          code === 90 ||
          code === 95 ||
          code === 100;
  const updateInputs = (state: HabitatConnectivityOptions) => {
    const key = `${state.habitat}/${state.resistance}`;
    if (key === lastStateKey) return;
    lastStateKey = key;
    revision++;
    const mask = new Uint32Array(cellCount);
    for (let index = 0; index < cellCount; index++) {
      mask[index] = Number(coverSelected(classes[index], state.habitat));
    }
    const friction = new Float32Array(cellCount);
    for (let index = 0; index < cellCount; index++) {
      const code = classes[index];
      friction[index] =
        state.resistance === 'mild' ? (MILD_RESISTANCE[code] ?? 3) : (STRONG_RESISTANCE[code] ?? 4);
    }
    buffers.mask.write(mask);
    buffers.friction.write(friction);
    buffers.revision.write(new Uint32Array([revision]));
    dirty = true;
    summaryReader.markStale();
  };
  updateInputs(ctx.options as HabitatConnectivityOptions);

  const getCompiledGraphs = (): readonly CompiledGPUCommandGraph<void>[] => [compiled];
  const getLayers = (): Layer[] => {
    const state = ctx.options;
    const coreValues =
      state.coreCells === 0 ? buffers.mask : buffers.coreVariants[state.coreCells - 1];
    const values =
      state.layer === 'cover'
        ? buffers.cover
        : state.layer === 'patches'
          ? buffers.labels
          : state.layer === 'cost'
            ? buffers.costs
            : state.layer === 'core'
              ? coreValues
              : buffers.mask;
    if (state.layer === 'cover') {
      return [
        new UtmRasterLayer({
          id: 'habitat-worldcover',
          frame,
          values,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: WORLD_COVER_CLASSES.map(({color}) => color),
          opacity: state.opacity
        })
      ];
    }
    if (state.layer === 'mask') {
      return [
        new UtmRasterLayer({
          id: 'habitat-mask-layer',
          frame,
          values,
          valueFormat: 'uint32',
          colormap: 'mask',
          color: [48, 190, 112, 235],
          opacity: state.opacity
        })
      ];
    }
    if (state.layer === 'patches') {
      return [
        new UtmRasterLayer({
          id: 'habitat-patches-layer',
          frame,
          values,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: HABITAT_PATCH_COLORS,
          noDataValue: 0,
          opacity: state.opacity
        })
      ];
    }
    if (state.layer === 'core') {
      return [
        new UtmRasterLayer({
          id: 'habitat-core-layer',
          frame,
          values,
          valueFormat: 'uint32',
          colormap: 'mask',
          color: [255, 214, 70, 245],
          opacity: state.opacity
        })
      ];
    }
    if (state.layer === 'cost') {
      return [
        new UtmRasterLayer({
          id: 'habitat-cost-layer',
          frame,
          values,
          valueFormat: 'float32',
          colormap: 'cividis',
          valueRange: [0, 25000],
          noDataColor: [0, 0, 0, 0],
          opacity: state.opacity
        }),
        new UtmRasterLayer({
          id: 'habitat-cost-sources-layer',
          frame,
          values: buffers.proximityCore,
          valueFormat: 'uint32',
          colormap: 'mask',
          color: [255, 214, 70, 245],
          opacity: state.opacity
        })
      ];
    }
    return [];
  };

  const instance: SceneInstance<HabitatConnectivityOptions> = {
    getCompiledGraphs,
    getLayers,
    setOption(id) {
      if (id === 'layer' || id === 'opacity') ctx.requestLayers();
      else {
        if (id === 'habitat') {
          updateInputs(ctx.options as HabitatConnectivityOptions);
          ctx.setReadout('patchCount', 'updating selected land-cover patches…');
          ctx.setReadout('patchPerimeter', 'updating selected land-cover patches…');
        }
        dirty = true;
        ctx.requestLayers();
      }
    },
    encode(commandEncoder) {
      if (destroyed || ctx.signal.aborted || !isCurrent()) return;
      updateInputs(ctx.options as HabitatConnectivityOptions);
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        summaryReader.request(commandEncoder);
      }
      summaryReader.flush(commandEncoder);
    },
    destroy() {
      destroyed = true;
      resources.destroy();
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [longitude, latitude] = event.coordinate;
      const [column, row] = frame.lngLatToCell(longitude, latitude);
      if (column < 0 || row < 0 || column >= GRID_SIZE || row >= GRID_SIZE) return null;
      const index = Math.floor(row) * GRID_SIZE + Math.floor(column);
      return `WorldCover ${classes[index]} · ${ctx.options.layer === 'cost' ? 'relative modeled walking cost' : '20 m cell'}`;
    }
  };
  return instance;
}

/** Rebuild only the owned compute instance; the controller never retains released buffers. */
export async function createHabitatConnectivity(
  ctx: SceneContext<HabitatConnectivityOptions>
): Promise<SceneInstance<HabitatConnectivityOptions>> {
  const graphSwap = new GraphSwap();
  let generation = 0;
  let disposed = false;
  let rebuilding = false;
  const buildInstance = () => {
    const instanceGeneration = generation;
    return createHabitatConnectivityInstance(
      ctx,
      () => !disposed && generation === instanceGeneration
    );
  };
  let activeInstance = await buildInstance();
  return {
    getCompiledGraphs: () => activeInstance.getCompiledGraphs(),
    getLayers: () => (rebuilding ? [] : activeInstance.getLayers()),
    encode(commandEncoder, currentFrame) {
      if (!rebuilding && !disposed) activeInstance.encode(commandEncoder, currentFrame);
    },
    setOption(id, value, state) {
      if (id === 'countRasterBorder') {
        rebuilding = true;
        const requestGeneration = ++generation;
        ctx.setReadout('patchPerimeter', 'rebuilding patch metrics · previous summary cleared');
        ctx.requestLayers();
        void graphSwap
          .request({
            build: buildInstance,
            commit(nextInstance) {
              const previousInstance = activeInstance;
              activeInstance = nextInstance;
              rebuilding = false;
              previousInstance.destroy();
              ctx.setStatus('');
              ctx.requestLayers();
            },
            discard(nextInstance) {
              nextInstance.destroy();
            }
          })
          .catch(error => {
            if (disposed || requestGeneration !== generation) return;
            ctx.setStatus(`Patch-metrics graph rebuild failed: ${String(error)}`);
            ctx.requestLayers();
          });
        return;
      }
      activeInstance.setOption?.(id, value, state);
    },
    getTooltip: event => (rebuilding ? null : (activeInstance.getTooltip?.(event) ?? null)),
    destroy() {
      disposed = true;
      generation++;
      graphSwap.destroy();
      activeInstance.destroy();
    }
  };
}
