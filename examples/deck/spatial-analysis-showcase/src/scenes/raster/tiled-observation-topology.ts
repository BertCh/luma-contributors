// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  addRasterRegionGraphTiles,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {
  GPURasterConnectedComponents,
  GPURasterCrossTileComponents,
  GPURasterDenseComponents,
  GPURasterRegionMeasurements,
  type GPURasterRegionMeasurementOutputs
} from '@luma.gl/graph/gpu-raster';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {addKernelPass} from '../../engine/mode-kernels';

const CROP_SIZE = 128;
const FOCUS_OFFSET = 32;
const FOCUS_SIZE = 64;
const OWNED_CORE_SIZE = 32;
const OWNED_CORE_COUNT = 4;

export const TILED_TOPOLOGY_THRESHOLD = 0.2;
export const TILED_TOPOLOGY_REGION_CAPACITY = 128;
export const TILED_TOPOLOGY_GLOBAL_REGION_COUNT = TILED_TOPOLOGY_REGION_CAPACITY * 2;
export const TILED_TOPOLOGY_CELL_COUNT = OWNED_CORE_SIZE * OWNED_CORE_SIZE;

export type TiledObservationSource = {values: Float32Array; validity: Uint32Array};

export type TiledObservationTopologyTile = {
  index: number;
  origin: readonly [number, number];
  values: Float32Array;
  validity: Uint32Array;
};

type TopologyViewFactory = {
  float(name: string, buffer: Buffer, length?: number): GraphDataView<'float32'>;
  word(name: string, buffer: Buffer, length?: number): GraphDataView<'uint32'>;
};

function makeTopologyViewFactory(views: TopologyViewFactory): TopologyViewFactory {
  return {
    float(name, buffer, length = buffer.byteLength / 4) {
      return views.float(name, buffer, length);
    },
    word(name, buffer, length = buffer.byteLength / 4) {
      return views.word(name, buffer, length);
    }
  };
}

type TopologyMeasurements = {
  pixelCounts: GraphDataView<'uint32'>;
  intensityCounts: GraphDataView<'uint32'>;
  intensitySums: GraphDataView<'float32'>;
  intensityMinimums: GraphDataView<'float32'>;
  intensityMaximums: GraphDataView<'float32'>;
  intensityMeans: GraphDataView<'float32'>;
  columnSums: GraphDataView<'float32'>;
  rowSums: GraphDataView<'float32'>;
  centroidColumns: GraphDataView<'float32'>;
  centroidRows: GraphDataView<'float32'>;
  areas: GraphDataView<'float32'>;
};

type TopologyClassBuffers = {
  mask: Buffer;
  sparseLabels: Buffer;
  sparseValidity: Buffer;
  localConverged: Buffer;
  localIterations: Buffer;
  denseLabels: Buffer;
  denseValidity: Buffer;
  localCount: Buffer;
  localRequiredCount: Buffer;
  localOverflow: Buffer;
  localMeasurements: TopologyMeasurements;
  globalLabels: Buffer;
  globalValidity: Buffer;
  globalMeasurements: TopologyMeasurements;
  globalPopulation: Buffer;
  globalCount: Buffer;
  globalRequiredCount: Buffer;
  globalConverged: Buffer;
  globalOverflow: Buffer;
};

export type TiledObservationTopologyGraph = {
  display: Buffer;
  low: TopologyClassBuffers[];
  high: TopologyClassBuffers[];
  tileValues: Buffer[];
  tileValidity: Buffer[];
  stableLabels: Buffer[];
  regionAdjacency: Buffer;
  regionWeights: Buffer;
  regionMapping: Buffer;
  regionStatus: Buffer;
  summaryBuffers: Buffer[];
  summaryByteLength: number;
  regionAdjacencyBytes: number;
  cellCount: number;
};

function createMeasurements(
  resources: SpatialAnalysisResources,
  views: TopologyViewFactory,
  id: string,
  capacity: number,
  pixelCountsBuffer?: Buffer
): TopologyMeasurements {
  const words = (name: string) =>
    views.word(name, resources.createBuffer(name, capacity * 4), capacity);
  const floats = (name: string) =>
    views.float(name, resources.createBuffer(name, capacity * 4), capacity);
  return {
    pixelCounts: views.word(
      `${id}-pixel-counts-view`,
      pixelCountsBuffer ?? resources.createBuffer(`${id}-pixel-counts`, capacity * 4),
      capacity
    ),
    intensityCounts: words(`${id}-intensity-counts`),
    intensitySums: floats(`${id}-intensity-sums`),
    intensityMinimums: floats(`${id}-intensity-minimums`),
    intensityMaximums: floats(`${id}-intensity-maximums`),
    intensityMeans: floats(`${id}-intensity-means`),
    columnSums: floats(`${id}-column-sums`),
    rowSums: floats(`${id}-row-sums`),
    centroidColumns: floats(`${id}-centroid-columns`),
    centroidRows: floats(`${id}-centroid-rows`),
    areas: floats(`${id}-areas`)
  };
}

function asMeasurementOutputs(
  measurements: TopologyMeasurements
): GPURasterRegionMeasurementOutputs {
  return measurements;
}

function createClassBuffers(
  resources: SpatialAnalysisResources,
  views: TopologyViewFactory,
  id: string
): TopologyClassBuffers {
  const pixels = TILED_TOPOLOGY_CELL_COUNT;
  const capacity = TILED_TOPOLOGY_REGION_CAPACITY;
  const globalPopulation = resources.createBuffer(
    `${id}-global-pixel-counts`,
    TILED_TOPOLOGY_REGION_CAPACITY * 4
  );
  const globalMeasurements = createMeasurements(
    resources,
    views,
    `${id}-global`,
    capacity,
    globalPopulation
  );
  return {
    mask: resources.createBuffer(`${id}-mask`, pixels * 4),
    sparseLabels: resources.createBuffer(`${id}-sparse-labels`, pixels * 4),
    sparseValidity: resources.createBuffer(`${id}-sparse-validity`, pixels * 4),
    localConverged: resources.createBuffer(`${id}-local-converged`, 4),
    localIterations: resources.createBuffer(`${id}-local-iterations`, 4),
    denseLabels: resources.createBuffer(`${id}-dense-labels`, pixels * 4),
    denseValidity: resources.createBuffer(`${id}-dense-validity`, pixels * 4),
    localCount: resources.createBuffer(`${id}-local-count`, 4),
    localRequiredCount: resources.createBuffer(`${id}-local-required-count`, 4),
    localOverflow: resources.createBuffer(`${id}-local-overflow`, 4),
    localMeasurements: createMeasurements(resources, views, `${id}-local`, capacity),
    globalLabels: resources.createBuffer(`${id}-global-labels`, pixels * 4),
    globalValidity: resources.createBuffer(`${id}-global-validity`, pixels * 4),
    globalMeasurements,
    globalPopulation,
    globalCount: resources.createBuffer(`${id}-global-count`, 4),
    globalRequiredCount: resources.createBuffer(`${id}-global-required-count`, 4),
    globalConverged: resources.createBuffer(`${id}-global-converged`, 4),
    globalOverflow: resources.createBuffer(`${id}-global-overflow`, 4)
  };
}

/** Adds two independently reconciled classes and the source-region adjacency graph. */
export function addTiledObservationTopology<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  resources: SpatialAnalysisResources,
  views: TopologyViewFactory,
  source: TiledObservationSource
): TiledObservationTopologyGraph {
  const topologyViews = makeTopologyViewFactory(views);
  const tiles = makeTiledObservationTopologyTiles(source);
  const localMetadata = makeGPURasterMetadata({
    width: 64,
    height: 64,
    cellSize: 60,
    origin: [0, FOCUS_SIZE * 60]
  });
  const tileMetadata = tiles.map(tile =>
    makeGPURasterMetadata({
      width: OWNED_CORE_SIZE,
      height: OWNED_CORE_SIZE,
      cellSize: 60,
      origin: [tile.origin[0] * 60, (FOCUS_SIZE - tile.origin[1]) * 60]
    })
  );
  const tileValues = tiles.map(tile =>
    resources.createBuffer(`topology-tile-${tile.index}-nir`, tile.values)
  );
  const tileValidity = tiles.map(tile =>
    resources.createBuffer(`topology-tile-${tile.index}-validity`, tile.validity)
  );
  const low = tiles.map(tile =>
    createClassBuffers(resources, topologyViews, `topology-tile-${tile.index}-low`)
  );
  const high = tiles.map(tile =>
    createClassBuffers(resources, topologyViews, `topology-tile-${tile.index}-high`)
  );
  const topologyDisplay = resources.createBuffer(
    'topology-focus-class-display',
    new Uint32Array(CROP_SIZE * CROP_SIZE).fill(0xffffffff)
  );
  const cellCount = TILED_TOPOLOGY_GLOBAL_REGION_COUNT;
  const regionAdjacency = resources.createBuffer(
    'topology-region-adjacency',
    cellCount * cellCount * 4
  );
  const regionWeights = resources.createBuffer(
    'topology-region-weights',
    cellCount * cellCount * 4
  );
  const regionMapping = resources.createBuffer('topology-region-mapping', cellCount * 4);
  const regionStatus = resources.createBuffer('topology-region-status', 4);

  for (const [tileIndex] of tiles.entries()) {
    const sourceValues = topologyViews.float(
      `topology-tile-${tileIndex}-nir-view`,
      tileValues[tileIndex]!
    );
    const sourceValidity = topologyViews.word(
      `topology-tile-${tileIndex}-validity-view`,
      tileValidity[tileIndex]!
    );
    const lowMask = topologyViews.word(
      `topology-tile-${tileIndex}-low-mask-view`,
      low[tileIndex]!.mask
    );
    const highMask = topologyViews.word(
      `topology-tile-${tileIndex}-high-mask-view`,
      high[tileIndex]!.mask
    );
    addKernelPass(graph, {
      id: `topology-tile-${tileIndex}-split-nir-at-${TILED_TOPOLOGY_THRESHOLD}`,
      invocationCount: TILED_TOPOLOGY_CELL_COUNT,
      bindings: [
        {name: 'values', view: sourceValues, type: 'f32', access: 'read'},
        {name: 'validity', view: sourceValidity, type: 'u32', access: 'read'},
        {name: 'lowMask', view: lowMask, type: 'u32', access: 'read_write'},
        {name: 'highMask', view: highMask, type: 'u32', access: 'read_write'}
      ],
      body: `let value = values[valuesOffset + index];
let valid = validity[validityOffset + index] != 0u && value == value && abs(value) < 3.402823e38;
lowMask[lowMaskOffset + index] = select(0u, 1u, valid && value < ${TILED_TOPOLOGY_THRESHOLD});
highMask[highMaskOffset + index] = select(0u, 1u, valid && value >= ${TILED_TOPOLOGY_THRESHOLD});`
    });
    for (const [className, classBuffers] of [
      ['low', low[tileIndex]!],
      ['high', high[tileIndex]!]
    ] as const) {
      const classInput = topologyViews.word(
        `topology-tile-${tileIndex}-${className}-mask-input`,
        classBuffers.mask
      );
      const sparse = topologyViews.word(
        `topology-tile-${tileIndex}-${className}-sparse-view`,
        classBuffers.sparseLabels
      );
      const sparseValidity = topologyViews.word(
        `topology-tile-${tileIndex}-${className}-sparse-validity-view`,
        classBuffers.sparseValidity
      );
      const converged = topologyViews.word(
        `topology-tile-${tileIndex}-${className}-converged-view`,
        classBuffers.localConverged,
        1
      );
      const dense = topologyViews.word(
        `topology-tile-${tileIndex}-${className}-dense-view`,
        classBuffers.denseLabels
      );
      const denseValidity = topologyViews.word(
        `topology-tile-${tileIndex}-${className}-dense-validity-view`,
        classBuffers.denseValidity
      );
      const componentCount = topologyViews.word(
        `topology-tile-${tileIndex}-${className}-component-count-view`,
        classBuffers.localCount,
        1
      );
      const requiredCount = topologyViews.word(
        `topology-tile-${tileIndex}-${className}-required-count-view`,
        classBuffers.localRequiredCount,
        1
      );
      const overflow = topologyViews.word(
        `topology-tile-${tileIndex}-${className}-overflow-view`,
        classBuffers.localOverflow,
        1
      );
      new GPURasterConnectedComponents({
        id: `topology-tile-${tileIndex}-${className}-connected-components`,
        width: OWNED_CORE_SIZE,
        height: OWNED_CORE_SIZE,
        input: {
          id: `topology-tile-${tileIndex}-${className}-mask`,
          format: 'uint32',
          storage: {kind: 'buffer', values: classInput},
          validity: sourceValidity
        },
        output: sparse,
        outputValidity: sparseValidity,
        converged,
        iterationCount: topologyViews.word(
          `topology-tile-${tileIndex}-${className}-iteration-view`,
          classBuffers.localIterations,
          1
        ),
        connectivity: 8
      }).addToGraph(graph);
      new GPURasterDenseComponents({
        id: `topology-tile-${tileIndex}-${className}-dense-components`,
        width: OWNED_CORE_SIZE,
        height: OWNED_CORE_SIZE,
        input: sparse,
        inputValidity: sparseValidity,
        converged,
        output: dense,
        outputValidity: denseValidity,
        componentCount,
        requiredComponentCount: requiredCount,
        overflow,
        capacity: TILED_TOPOLOGY_REGION_CAPACITY
      }).addToGraph(graph);
      new GPURasterRegionMeasurements({
        id: `topology-tile-${tileIndex}-${className}-measurements`,
        metadata: tileMetadata[tileIndex]!,
        labels: dense,
        labelValidity: denseValidity,
        converged,
        componentCount,
        overflow,
        intensity: {
          id: `topology-tile-${tileIndex}-nir-intensity`,
          format: 'float32',
          storage: {kind: 'buffer', values: sourceValues},
          validity: sourceValidity
        },
        output: asMeasurementOutputs(classBuffers.localMeasurements),
        capacity: TILED_TOPOLOGY_REGION_CAPACITY
      }).addToGraph(graph);
    }
  }

  const globalClassOutputs = [
    {name: 'low', buffers: low},
    {name: 'high', buffers: high}
  ] as const;
  for (const {name, buffers} of globalClassOutputs) {
    const componentTiles = tiles.map((tile, tileIndex) => {
      const classBuffers = buffers[tileIndex]!;
      const local = classBuffers.localMeasurements;
      return {
        metadata: tileMetadata[tileIndex]!,
        pixelBounds: [
          tile.origin[0],
          tile.origin[1],
          tile.origin[0] + OWNED_CORE_SIZE,
          tile.origin[1] + OWNED_CORE_SIZE
        ] as const,
        labels: topologyViews.word(
          `topology-${name}-tile-${tileIndex}-dense-input`,
          classBuffers.denseLabels
        ),
        labelValidity: topologyViews.word(
          `topology-${name}-tile-${tileIndex}-dense-validity-input`,
          classBuffers.denseValidity
        ),
        componentCount: topologyViews.word(
          `topology-${name}-tile-${tileIndex}-local-count-input`,
          classBuffers.localCount,
          1
        ),
        converged: topologyViews.word(
          `topology-${name}-tile-${tileIndex}-local-converged-input`,
          classBuffers.localConverged,
          1
        ),
        overflow: topologyViews.word(
          `topology-${name}-tile-${tileIndex}-local-overflow-input`,
          classBuffers.localOverflow,
          1
        ),
        measurements: asMeasurementOutputs(local),
        outputLabels: topologyViews.word(
          `topology-${name}-tile-${tileIndex}-global-label-output`,
          classBuffers.globalLabels
        ),
        outputValidity: topologyViews.word(
          `topology-${name}-tile-${tileIndex}-global-validity-output`,
          classBuffers.globalValidity
        )
      };
    });
    new GPURasterCrossTileComponents({
      id: `topology-${name}-cross-tile-components`,
      metadata: localMetadata,
      tiles: componentTiles,
      connectivity: 8,
      maximumIterations: 32,
      componentCount: topologyViews.word(
        `topology-${name}-global-count`,
        buffers[0]!.globalCount,
        1
      ),
      requiredComponentCount: topologyViews.word(
        `topology-${name}-global-required-count`,
        buffers[0]!.globalRequiredCount,
        1
      ),
      converged: topologyViews.word(
        `topology-${name}-global-converged`,
        buffers[0]!.globalConverged,
        1
      ),
      overflow: topologyViews.word(
        `topology-${name}-global-overflow`,
        buffers[0]!.globalOverflow,
        1
      ),
      output: asMeasurementOutputs(buffers[0]!.globalMeasurements),
      capacity: TILED_TOPOLOGY_REGION_CAPACITY
    }).addToGraph(graph);
  }

  const stableLabels = tiles.map(tile =>
    resources.createBuffer(
      `topology-tile-${tile.index}-stable-region-labels`,
      TILED_TOPOLOGY_CELL_COUNT * 4
    )
  );
  const displayView = topologyViews.word(
    'topology-focus-class-display-view',
    topologyDisplay,
    CROP_SIZE * CROP_SIZE
  );
  for (const [tileIndex, tile] of tiles.entries()) {
    const lowLabels = topologyViews.word(
      `topology-tile-${tileIndex}-low-global-labels-for-combine`,
      low[tileIndex]!.globalLabels
    );
    const lowValidity = topologyViews.word(
      `topology-tile-${tileIndex}-low-global-validity-for-combine`,
      low[tileIndex]!.globalValidity
    );
    const highLabels = topologyViews.word(
      `topology-tile-${tileIndex}-high-global-labels-for-combine`,
      high[tileIndex]!.globalLabels
    );
    const highValidity = topologyViews.word(
      `topology-tile-${tileIndex}-high-global-validity-for-combine`,
      high[tileIndex]!.globalValidity
    );
    const stableLabelView = topologyViews.word(
      `topology-tile-${tileIndex}-stable-labels-output`,
      stableLabels[tileIndex]!
    );
    const displayOffset =
      (FOCUS_OFFSET + tile.origin[1]) * CROP_SIZE + FOCUS_OFFSET + tile.origin[0];
    addKernelPass(graph, {
      id: `topology-tile-${tileIndex}-combine-stable-labels-and-display`,
      invocationCount: TILED_TOPOLOGY_CELL_COUNT,
      bindings: [
        {name: 'lowLabels', view: lowLabels, type: 'u32', access: 'read'},
        {name: 'lowValidity', view: lowValidity, type: 'u32', access: 'read'},
        {name: 'highLabels', view: highLabels, type: 'u32', access: 'read'},
        {name: 'highValidity', view: highValidity, type: 'u32', access: 'read'},
        {name: 'stableLabels', view: stableLabelView, type: 'u32', access: 'read_write'},
        {name: 'displayValues', view: displayView, type: 'u32', access: 'read_write'}
      ],
      body: `let lowLabel = lowLabels[lowLabelsOffset + index];
let highLabel = highLabels[highLabelsOffset + index];
let valid = lowValidity[lowValidityOffset + index] != 0u && highValidity[highValidityOffset + index] != 0u;
let region = select(select(0xffffffffu, lowLabel - 1u, lowLabel != 0u), 128u + highLabel - 1u, lowLabel == 0u && highLabel != 0u);
stableLabels[stableLabelsOffset + index] = select(0xffffffffu, region, valid);
displayValues[displayValuesOffset + ${displayOffset}u + (index / ${OWNED_CORE_SIZE}u) * ${CROP_SIZE}u + index % ${OWNED_CORE_SIZE}u] = select(0xffffffffu, region, valid && region != 0xffffffffu);`
    });
  }

  const regionAdjacencyView = topologyViews.word(
    'topology-region-adjacency-input',
    regionAdjacency,
    cellCount * cellCount
  );
  const regionWeightsView = topologyViews.float(
    'topology-region-weights-input',
    regionWeights,
    cellCount * cellCount
  );
  const regionMappingView = topologyViews.word(
    'topology-region-mapping-output',
    regionMapping,
    cellCount
  );
  const regionStatusView = topologyViews.word('topology-region-status-output', regionStatus, 1);
  const regionTiles = tiles.map((tile, tileIndex) => ({
    grid: tileMetadata[tileIndex]!,
    origin: tile.origin,
    labels: topologyViews.word(
      `topology-region-graph-labels-${tileIndex}`,
      stableLabels[tileIndex]!
    ),
    values: topologyViews.float(`topology-region-graph-values-${tileIndex}`, tileValues[tileIndex]!)
  }));
  addRasterRegionGraphTiles(graph, {
    id: 'topology-region-adjacency-graph',
    grid: localMetadata,
    tiles: regionTiles,
    regionCount: TILED_TOPOLOGY_GLOBAL_REGION_COUNT,
    mergeMode: 'fixed',
    mergeThreshold: 0,
    maximumIterations: 32,
    output: {
      adjacency: regionAdjacencyView,
      weights: regionWeightsView,
      mapping: regionMappingView,
      status: {iterationLimitReached: regionStatusView}
    }
  });

  const summaryBuffers = [
    ...low.flatMap(classBuffers => [
      classBuffers.localCount,
      classBuffers.localRequiredCount,
      classBuffers.localConverged,
      classBuffers.localOverflow
    ]),
    ...high.flatMap(classBuffers => [
      classBuffers.localCount,
      classBuffers.localRequiredCount,
      classBuffers.localConverged,
      classBuffers.localOverflow
    ]),
    low[0]!.globalCount,
    low[0]!.globalRequiredCount,
    low[0]!.globalConverged,
    low[0]!.globalOverflow,
    low[0]!.globalPopulation,
    high[0]!.globalCount,
    high[0]!.globalRequiredCount,
    high[0]!.globalConverged,
    high[0]!.globalOverflow,
    high[0]!.globalPopulation,
    regionAdjacency,
    regionMapping,
    regionStatus
  ];
  const summaryByteLength = summaryBuffers.reduce((total, buffer) => total + buffer.byteLength, 0);
  return {
    display: topologyDisplay,
    low,
    high,
    tileValues,
    tileValidity,
    stableLabels,
    regionAdjacency,
    regionWeights,
    regionMapping,
    regionStatus,
    summaryBuffers,
    summaryByteLength,
    regionAdjacencyBytes: regionAdjacency.byteLength,
    cellCount
  };
}

/** The central 64 × 64 area, divided into four disjoint 32 × 32 ownership cores. */
export function makeTiledObservationTopologyTiles(
  source: TiledObservationSource
): TiledObservationTopologyTile[] {
  const tiles: TiledObservationTopologyTile[] = [];
  for (let tileRow = 0; tileRow < 2; tileRow++) {
    for (let tileColumn = 0; tileColumn < 2; tileColumn++) {
      const values = new Float32Array(TILED_TOPOLOGY_CELL_COUNT).fill(Number.NaN);
      const validity = new Uint32Array(TILED_TOPOLOGY_CELL_COUNT);
      for (let row = 0; row < OWNED_CORE_SIZE; row++) {
        for (let column = 0; column < OWNED_CORE_SIZE; column++) {
          const sourceRow = FOCUS_OFFSET + tileRow * OWNED_CORE_SIZE + row;
          const sourceColumn = FOCUS_OFFSET + tileColumn * OWNED_CORE_SIZE + column;
          const sourceIndex = sourceRow * CROP_SIZE + sourceColumn;
          const targetIndex = row * OWNED_CORE_SIZE + column;
          values[targetIndex] = source.values[sourceIndex];
          validity[targetIndex] = source.validity[sourceIndex];
        }
      }
      tiles.push({
        index: tileRow * 2 + tileColumn,
        origin: [tileColumn * OWNED_CORE_SIZE, tileRow * OWNED_CORE_SIZE],
        values,
        validity
      });
    }
  }
  return tiles;
}

export function getTiledObservationTopologyFocusBounds(): readonly [
  number,
  number,
  number,
  number
] {
  return [FOCUS_OFFSET, FOCUS_OFFSET, FOCUS_OFFSET + FOCUS_SIZE, FOCUS_OFFSET + FOCUS_SIZE];
}

export function getTiledObservationTopologyTileSize(): number {
  return OWNED_CORE_SIZE;
}

export function getTiledObservationTopologyFocusSize(): number {
  return FOCUS_SIZE;
}

export function getTiledObservationTopologyTileCount(): number {
  return OWNED_CORE_COUNT;
}
