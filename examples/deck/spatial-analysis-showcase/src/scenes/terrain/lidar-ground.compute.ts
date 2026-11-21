// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  GPUCommandGraph,
  DrawCommandBuffer,
  GPUCompaction,
  createTransientView,
  GPUElementwise,
  GPUGather,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {
  getRasterMetadataFromPointCloudGrid,
  GPUPointCloudFeatures,
  GPUPointCloudCoverageValidity,
  GPUPointCloudGround,
  GPUPointCloudHeight,
  GPUPointCloudNeighborhoods,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudSurfaces
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {GPUPointCloudVoxelDownsample} from '@luma.gl/graph/gpu-pointcloud';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import type {LidarGroundOptions} from './lidar-ground.scene';
import {
  createGroundSeedCoverageNode,
  createFeatureDisplayClassesNode,
  addLidarFeatureSummary,
  getLidarFilterSupportText,
  createSupportedGroundClassesNode,
  createGroundDisplayClassesNode,
  LIDAR_GROUND_DISPLAY_CATEGORIES,
  LIDAR_FEATURE_DISPLAY_CATEGORIES,
  LIDAR_GROUND_GRID,
  LIDAR_GROUND_RADII
} from './point-cloud-ground-support';

const RASTER_GRID = LIDAR_GROUND_GRID;
const NEIGHBOR_CAPACITY = 16;
const VOXEL_SIZE = 2;
const VOXEL_GRID_SIZE = [110, 110, 45] as const;
const VOXEL_COUNT = VOXEL_GRID_SIZE.reduce((product, size) => product * size, 1);
const CLASS_PALETTE = [
  [232, 174, 94, 255],
  [146, 206, 164, 255],
  [229, 105, 99, 255],
  [92, 156, 215, 255],
  [223, 98, 94, 255],
  [185, 183, 175, 255],
  [150, 155, 160, 255]
] as const;

export async function createLidarGround(
  ctx: SceneContext<LidarGroundOptions>
): Promise<SceneInstance<LidarGroundOptions>> {
  const dataset = ctx.datasets.get('lidar-ground');
  const count = dataset.count;
  const projected = dataset.projectColumn('position');
  const elevation = dataset.column<Float32Array>('elevation');
  const sourceClasses = dataset.column<Uint32Array>('classification');
  const positions = new Float32Array(count * 3);
  const positions2D = new Float32Array(count * 2);
  const displayClasses = new Uint32Array(count);
  const classSlot = new Map([
    [1, 0],
    [2, 1],
    [7, 2],
    [9, 3],
    [18, 4],
    [20, 5]
  ]);
  for (let row = 0; row < count; row++) {
    positions[row * 3] = projected[row * 2];
    positions[row * 3 + 1] = projected[row * 2 + 1];
    positions[row * 3 + 2] = elevation[row];
    positions2D[row * 2] = projected[row * 2];
    positions2D[row * 2 + 1] = projected[row * 2 + 1];
    displayClasses[row] = classSlot.get(sourceClasses[row]) ?? 6;
  }

  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'lidar-ground');
  const positionBuffer = resources.createBuffer('positions', positions);
  const position2DBuffer = resources.createBuffer('positions-2d', positions2D);
  const sourceClassBuffer = resources.createBuffer('source-classes', sourceClasses);
  const displayClassBuffer = resources.createBuffer('display-classes', displayClasses);
  const surfaceCount = RASTER_GRID.width * RASTER_GRID.height;
  const neighborhoodIndicesBuffer = resources.createBuffer(
    'neighborhood-indices',
    count * NEIGHBOR_CAPACITY * 4
  );
  const neighborhoodDistanceBuffer = resources.createBuffer(
    'neighborhood-distances',
    count * NEIGHBOR_CAPACITY * 4
  );
  const neighborhoodCountBuffer = resources.createBuffer('neighborhood-counts', count * 4);
  const neighborhoodStatusBuffer = resources.createBuffer('neighborhood-status', count * 4);
  const meanDistanceBuffer = resources.createBuffer('mean-distance', count * 4);
  const inlierBuffer = resources.createBuffer('inlier-mask', count * 4);
  const firstTerrainBuffer = resources.createBuffer('first-terrain', surfaceCount * 4);
  const firstSurfaceBuffer = resources.createBuffer('first-surface', surfaceCount * 4);
  const firstCanopyBuffer = resources.createBuffer('first-canopy', surfaceCount * 4);
  const firstCoverageBuffer = resources.createBuffer('first-coverage', surfaceCount * 4);
  const groundSeedCoverageBuffer = resources.createBuffer('ground-seed-coverage', surfaceCount * 4);
  const supportedClassificationsBuffer = resources.createBuffer(
    'supported-classifications',
    count * 4
  );
  const groundMaskBuffer = resources.createBuffer('ground-mask', count * 4);
  const groundDisplayBuffer = resources.createBuffer('ground-display', count * 4);
  const classifiedBuffer = resources.createBuffer('classified', count * 4);
  const groundStatusBuffer = resources.createBuffer('ground-status', count * 4);
  const terrainBuffer = resources.createBuffer('terrain', surfaceCount * 4);
  const surfaceBuffer = resources.createBuffer('surface', surfaceCount * 4);
  const canopyBuffer = resources.createBuffer('canopy', surfaceCount * 4);
  const coverageBuffer = resources.createBuffer('coverage', surfaceCount * 4);
  const heightBuffer = resources.createBuffer('height', count * 4);
  const heightStatusBuffer = resources.createBuffer('height-status', count * 4);
  const normalsBuffer = resources.createBuffer('normals', count * 12);
  const eigenvaluesBuffer = resources.createBuffer('eigenvalues', count * 12);
  const shapesBuffer = resources.createBuffer('shapes', count * 12);
  const featureStatusBuffer = resources.createBuffer('feature-status', count * 4);
  const featureSummaryBuffer = resources.createBuffer('feature-summary', 20);
  const featureDisplayBuffer = resources.createBuffer('feature-display', count * 4);
  const thresholdCompleteBuffer = resources.createBuffer('threshold-complete', 4);
  const representativeBuffer = resources.createBuffer('voxel-representatives', VOXEL_COUNT * 4);
  const voxelCountsBuffer = resources.createBuffer('voxel-counts', VOXEL_COUNT * 4);
  const voxelFlagOnesBuffer = resources.createBuffer(
    'voxel-flag-ones',
    new Uint32Array(VOXEL_COUNT).fill(1)
  );
  const occupiedVoxelFlagsBuffer = resources.createBuffer('occupied-voxel-flags', VOXEL_COUNT * 4);
  const voxelPositionsBuffer = resources.createBuffer('voxel-positions', VOXEL_COUNT * 12);
  const compactIndicesBuffer = resources.createBuffer('compact-indices', VOXEL_COUNT * 4);
  const compactPositionBuffer = resources.createBuffer('compact-positions', VOXEL_COUNT * 8);
  const compactClassBuffer = resources.createBuffer('compact-classes', VOXEL_COUNT * 4);
  const voxelDrawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'lidar-ground-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  const graph = new GPUCommandGraph<void>(device, {id: 'lidar-ground-analysis'});
  const positionsView = importGraphBuffer(graph, 'positions', positionBuffer, 'float32x3', count);
  const classesView = importGraphBuffer(
    graph,
    'source-classes',
    sourceClassBuffer,
    'uint32',
    count
  );
  const neighborhoodIndices = importGraphBuffer(
    graph,
    'neighborhood-indices',
    neighborhoodIndicesBuffer,
    'uint32',
    count * NEIGHBOR_CAPACITY
  );
  const neighborhoodDistances = importGraphBuffer(
    graph,
    'neighborhood-distances',
    neighborhoodDistanceBuffer,
    'float32',
    count * NEIGHBOR_CAPACITY
  );
  const neighborhoodCounts = importGraphBuffer(
    graph,
    'neighborhood-counts',
    neighborhoodCountBuffer,
    'uint32',
    count
  );
  const neighborhoodStatus = importGraphBuffer(
    graph,
    'neighborhood-status',
    neighborhoodStatusBuffer,
    'uint32',
    count
  );
  const meanDistances = importGraphBuffer(
    graph,
    'mean-distance',
    meanDistanceBuffer,
    'float32',
    count
  );
  const inliers = importGraphBuffer(graph, 'inlier-mask', inlierBuffer, 'uint32', count);
  const firstTerrain = importGraphBuffer(
    graph,
    'first-terrain',
    firstTerrainBuffer,
    'float32',
    surfaceCount
  );
  const firstSurface = importGraphBuffer(
    graph,
    'first-surface',
    firstSurfaceBuffer,
    'float32',
    surfaceCount
  );
  const firstCanopy = importGraphBuffer(
    graph,
    'first-canopy',
    firstCanopyBuffer,
    'float32',
    surfaceCount
  );
  const firstCoverage = importGraphBuffer(
    graph,
    'first-coverage',
    firstCoverageBuffer,
    'uint32',
    surfaceCount
  );
  const groundMask = importGraphBuffer(graph, 'ground-mask', groundMaskBuffer, 'uint32', count);
  const classified = importGraphBuffer(graph, 'classified', classifiedBuffer, 'uint32', count);
  const groundStatus = importGraphBuffer(
    graph,
    'ground-status',
    groundStatusBuffer,
    'uint32',
    count
  );
  const terrain = importGraphBuffer(graph, 'terrain', terrainBuffer, 'float32', surfaceCount);
  const surface = importGraphBuffer(graph, 'surface', surfaceBuffer, 'float32', surfaceCount);
  const canopy = importGraphBuffer(graph, 'canopy', canopyBuffer, 'float32', surfaceCount);
  const coverage = importGraphBuffer(graph, 'coverage', coverageBuffer, 'uint32', surfaceCount);
  const heights = importGraphBuffer(graph, 'height', heightBuffer, 'float32', count);
  const heightStatus = importGraphBuffer(
    graph,
    'height-status',
    heightStatusBuffer,
    'uint32',
    count
  );
  const featureNormals = importGraphBuffer(graph, 'normals', normalsBuffer, 'float32x3', count);
  const featureEigenvalues = importGraphBuffer(
    graph,
    'eigenvalues',
    eigenvaluesBuffer,
    'float32x3',
    count
  );
  const featureShapes = importGraphBuffer(graph, 'shapes', shapesBuffer, 'float32x3', count);
  const featureStatus = importGraphBuffer(
    graph,
    'feature-status',
    featureStatusBuffer,
    'uint32',
    count
  );
  const representatives = importGraphBuffer(
    graph,
    'voxel-representatives',
    representativeBuffer,
    'uint32',
    VOXEL_COUNT
  );
  const voxelCounts = importGraphBuffer(
    graph,
    'voxel-counts',
    voxelCountsBuffer,
    'uint32',
    VOXEL_COUNT
  );
  const voxelFlagOnes = importGraphBuffer(
    graph,
    'voxel-flag-ones',
    voxelFlagOnesBuffer,
    'uint32',
    VOXEL_COUNT
  );
  const occupiedVoxelFlags = importGraphBuffer(
    graph,
    'occupied-voxel-flags',
    occupiedVoxelFlagsBuffer,
    'uint32',
    VOXEL_COUNT
  );
  const voxelPositions = importGraphBuffer(
    graph,
    'voxel-positions',
    voxelPositionsBuffer,
    'float32x3',
    VOXEL_COUNT
  );
  const compactIndices = importGraphBuffer(
    graph,
    'compact-indices',
    compactIndicesBuffer,
    'uint32',
    VOXEL_COUNT
  );
  const compactPositions = importGraphBuffer(
    graph,
    'compact-positions',
    compactPositionBuffer,
    'float32x2',
    VOXEL_COUNT
  );
  const compactClasses = importGraphBuffer(
    graph,
    'compact-classes',
    compactClassBuffer,
    'uint32',
    VOXEL_COUNT
  );
  const displayClassesView = importGraphBuffer(
    graph,
    'display-classes',
    displayClassBuffer,
    'uint32',
    count
  );
  const positions2DView = importGraphBuffer(
    graph,
    'positions-2d',
    position2DBuffer,
    'float32x2',
    count
  );
  const drawView = voxelDrawCommands.importToGraph(graph);
  const drawInstanceCount = graph.createDataView(drawView.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: Uint32Array.BYTES_PER_ELEMENT
  });

  graph.add(
    new GPUPointCloudNeighborhoods({
      id: 'lidar-ground-neighborhoods',
      positions: positionsView,
      radius: 2.5,
      capacity: NEIGHBOR_CAPACITY,
      grid: {origin: [-110, -110, -5], cellSize: 2.5, dimensions: [88, 88, 36]},
      output: {
        indices: neighborhoodIndices,
        squaredDistances: neighborhoodDistances,
        counts: neighborhoodCounts,
        status: neighborhoodStatus
      }
    })
  );
  graph.add(
    new GPUPointCloudStatisticalFilter({
      id: 'lidar-ground-statistical-filter',
      squaredDistances: neighborhoodDistances,
      counts: neighborhoodCounts,
      neighborhoodStatus,
      capacity: NEIGHBOR_CAPACITY,
      output: {
        meanDistances,
        validity: inliers,
        thresholdComplete: importGraphBuffer(
          graph,
          'threshold-complete',
          thresholdCompleteBuffer,
          'uint32',
          1
        )
      }
    })
  );
  graph.add(
    new GPUPointCloudSurfaces({
      id: 'lidar-ground-seed-surfaces',
      positions: positionsView,
      classifications: classesView,
      grid: RASTER_GRID,
      output: {
        terrain: firstTerrain,
        surface: firstSurface,
        canopy: firstCanopy,
        coverage: firstCoverage
      }
    })
  );
  const groundSeedCoverage = importGraphBuffer(
    graph,
    'ground-seed-coverage',
    groundSeedCoverageBuffer,
    'uint32',
    surfaceCount
  );
  const supportedClassifications = importGraphBuffer(
    graph,
    'supported-classifications',
    supportedClassificationsBuffer,
    'uint32',
    count
  );
  graph.add(createGroundSeedCoverageNode(graph, firstCoverage, groundSeedCoverage));
  graph.add(
    new GPUPointCloudGround({
      id: 'lidar-ground-pmf',
      positions: positionsView,
      classifications: classesView,
      grid: RASTER_GRID,
      radii: LIDAR_GROUND_RADII,
      initialThreshold: 0.3,
      slope: 0.1,
      maximumThreshold: 2,
      minimumSurface: firstTerrain,
      coverage: groundSeedCoverage,
      output: {groundMask, outputClassifications: classified, status: groundStatus}
    })
  );
  graph.add(
    createSupportedGroundClassesNode(graph, classified, groundStatus, supportedClassifications)
  );
  graph.add(
    createGroundDisplayClassesNode(
      graph,
      groundMask,
      groundStatus,
      importGraphBuffer(graph, 'ground-display', groundDisplayBuffer, 'uint32', count)
    )
  );
  graph.add(
    new GPUPointCloudSurfaces({
      id: 'lidar-ground-classified-surfaces',
      positions: positionsView,
      classifications: supportedClassifications,
      grid: RASTER_GRID,
      output: {terrain, surface, canopy, coverage}
    })
  );
  const terrainValidity = createTransientView(graph, 'terrain-validity', 'uint32', surfaceCount);
  graph.add(
    new GPUPointCloudCoverageValidity({
      id: 'lidar-ground-raster-validity',
      coverage,
      output: {
        surfaceValidity: createTransientView(graph, 'surface-validity', 'uint32', surfaceCount),
        terrainValidity,
        canopyValidity: createTransientView(graph, 'canopy-validity', 'uint32', surfaceCount)
      }
    })
  );
  const terrainBand = makeGPURasterBand(terrain, getRasterMetadataFromPointCloudGrid(RASTER_GRID), {
    id: 'lidar-ground-terrain',
    validity: terrainValidity
  });
  if (!terrainBand.validity) {
    throw new Error('lidar-ground terrain band must carry ground validity');
  }
  graph.add(
    new GPUPointCloudHeight({
      id: 'lidar-ground-height',
      positions: positionsView,
      grid: RASTER_GRID,
      terrain: terrainBand.storage.values,
      coverage: terrainBand.validity,
      coverageBit: 1,
      output: {heights, status: heightStatus}
    })
  );
  graph.add(
    new GPUPointCloudFeatures({
      id: 'lidar-ground-features',
      positions: positionsView,
      indices: neighborhoodIndices,
      counts: neighborhoodCounts,
      neighborhoodStatus,
      capacity: NEIGHBOR_CAPACITY,
      output: {
        normals: featureNormals,
        eigenvalues: featureEigenvalues,
        shapes: featureShapes,
        status: featureStatus
      }
    })
  );
  graph.add(
    createFeatureDisplayClassesNode(
      graph,
      featureStatus,
      importGraphBuffer(graph, 'feature-display', featureDisplayBuffer, 'uint32', count)
    )
  );
  graph.add(
    new GPUPointCloudVoxelDownsample({
      id: 'lidar-ground-voxel-downsample',
      positions: positionsView,
      origin: [-110, -110, -5],
      voxelSize: VOXEL_SIZE,
      gridSize: VOXEL_GRID_SIZE,
      representatives,
      counts: voxelCounts,
      output: voxelPositions
    })
  );
  graph.add(
    new GPUElementwise({
      id: 'lidar-ground-voxel-occupancy-flags',
      input: voxelCounts,
      inputB: voxelFlagOnes,
      output: occupiedVoxelFlags,
      operation: 'min'
    })
  );
  graph.add(
    new GPUCompaction({
      id: 'lidar-ground-compact-voxels',
      input: representatives,
      flags: occupiedVoxelFlags,
      output: compactIndices,
      count: drawInstanceCount
    })
  );
  graph.add(
    new GPUGather({
      id: 'lidar-ground-gather-voxel-positions',
      source: positions2DView,
      indices: compactIndices,
      output: compactPositions
    })
  );
  graph.add(
    new GPUGather({
      id: 'lidar-ground-gather-voxel-classes',
      source: displayClassesView,
      indices: compactIndices,
      output: compactClasses
    })
  );

  addLidarFeatureSummary(
    graph,
    featureStatus,
    importGraphBuffer(graph, 'feature-summary', featureSummaryBuffer, 'uint32', 5)
  );
  const compiled = resources.track(await graph.compileAsync());
  let analysisDirty = true;
  let filterThresholdComplete = false;
  const summaryReader = new SummaryReader(
    resources,
    'lidar-ground-voxel-count',
    [
      {buffer: voxelDrawCommands.buffer, size: 8},
      {buffer: featureSummaryBuffer, size: 20},
      {buffer: thresholdCompleteBuffer, size: 4}
    ],
    bytes => {
      ctx.setReadout('voxelPoints', new DataView(bytes).getUint32(4, true));
      const [incomplete, usableIncomplete, truncated, failed, degenerate] = new Uint32Array(
        bytes,
        8,
        5
      );
      ctx.setReadout(
        'featureSupport',
        `${incomplete} incomplete-support rows (${usableIncomplete} usable fits); ${truncated} truncated; ${failed} failed eigensolves; ${degenerate} degenerate. Halo completeness is not certified.`
      );
      filterThresholdComplete = new Uint32Array(bytes, 28, 1)[0] !== 0;
      ctx.setReadout('filterSupport', getLidarFilterSupportText(filterThresholdComplete));
      ctx.requestLayers();
    }
  );
  ctx.setReadout('samplePoints', count);
  ctx.setReadout(
    'featureSupport',
    'Feature diagnostics pending; halo completeness is not certified'
  );
  ctx.setReadout('filterSupport', 'Filter diagnostics pending');
  const origin = [dataset.defaultOrigin[0], dataset.defaultOrigin[1], 0] as [
    number,
    number,
    number
  ];
  return {
    getCompiledGraphs: () => [compiled] as CompiledGPUCommandGraph<never>[],
    setOption: () => ctx.requestLayers(),
    encode(commandEncoder) {
      if (analysisDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        summaryReader.request(commandEncoder);
        analysisDirty = false;
      } else summaryReader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const view = ctx.options.view;
      const base = {coordinateOrigin: origin, radiusPixels: ctx.options.pointSize, opacity: 0.9};
      if (view === 'voxels') {
        return [
          new SpatialAnalysisPointLayer({
            id: 'lidar-ground-voxel-representatives',
            positions: compactPositionBuffer,
            values: compactClassBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: CLASS_PALETTE,
            drawCommands: voxelDrawCommands,
            ...base
          })
        ];
      }
      if (view === 'outliers')
        return filterThresholdComplete
          ? [
              new SpatialAnalysisPointLayer({
                id: 'lidar-ground-statistical-support',
                positions: position2DBuffer,
                values: inlierBuffer,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: [
                  [204, 109, 97, 255],
                  [150, 211, 166, 255]
                ],
                instanceCount: count,
                ...base
              })
            ]
          : [];
      if (view === 'ground')
        return [
          new SpatialAnalysisPointLayer({
            id: 'lidar-ground-pmf-classification',
            positions: position2DBuffer,
            values: groundDisplayBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: LIDAR_GROUND_DISPLAY_CATEGORIES.map(({color}) => [...color, 255] as const),
            instanceCount: count,
            ...base
          })
        ];
      if (view === 'height')
        return [
          new SpatialAnalysisPointLayer({
            id: 'lidar-ground-relative-height',
            positions: position2DBuffer,
            values: heightBuffer,
            valueFormat: 'float32',
            colormap: 'cividis',
            valueRange: [0, 25],
            rampRange: [0.15, 1],
            instanceCount: count,
            ...base
          }),
          new SpatialAnalysisPointLayer({
            id: 'lidar-ground-height-support',
            positions: position2DBuffer,
            values: heightStatusBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [233, 181, 101, 240],
            noDataColor: [0, 0, 0, 0],
            instanceCount: count,
            ...base,
            radiusPixels: ctx.options.pointSize + 1
          })
        ];
      if (view === 'features')
        return [
          new SpatialAnalysisPointLayer({
            id: 'lidar-ground-feature-support',
            positions: position2DBuffer,
            values: featureDisplayBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: LIDAR_FEATURE_DISPLAY_CATEGORIES.map(({color}) => [...color, 255] as const),
            instanceCount: count,
            ...base
          })
        ];
      return [
        new SpatialAnalysisPointLayer({
          id: 'lidar-ground-source-classes',
          positions: position2DBuffer,
          values: displayClassBuffer,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: CLASS_PALETTE,
          instanceCount: count,
          ...base
        })
      ];
    },
    destroy: () => resources.destroy()
  };
}
