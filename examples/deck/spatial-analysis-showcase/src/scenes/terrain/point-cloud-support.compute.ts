// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUCompaction,
  GPUElementwise,
  GPUGather,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudOctree,
  GPUPointCloudRadiusFilter,
  GPUPointCloudVoxelDownsample
} from '@luma.gl/graph/gpu-pointcloud';
import {
  GPUPointCloudSMRF,
  GPUPointCloudVoxelMerge
} from '@luma-contributors/analysis/pointcloud-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import type {BarChartData} from '../chart-types';
import type {PointCloudSupportOptions} from './point-cloud-support.scene';
import {addPointCloudSupportVoxelPart} from './point-cloud-voxel-parts';

const TARGET_ROWS = 2000;
const VOXEL_SIZE_METERS = 6;
const CLASS_COLORS = [
  [232, 174, 94, 255],
  [146, 206, 164, 255],
  [229, 105, 99, 255],
  [92, 156, 215, 255],
  [223, 98, 94, 255],
  [185, 183, 175, 255],
  [150, 155, 160, 255]
] as const;
const OCTREE_COLORS = [
  [80, 145, 181, 255],
  [104, 171, 174, 255],
  [138, 185, 146, 255],
  [190, 190, 113, 255],
  [222, 165, 103, 255],
  [213, 133, 112, 255],
  [186, 118, 153, 255],
  [151, 127, 184, 255]
] as const;

type BuildSettings = Pick<
  PointCloudSupportOptions,
  'radius' | 'minimumNeighbors' | 'smrfWindow' | 'smrfSlope' | 'maximumDepth' | 'nodePointLimit'
>;
type AnalysisGraph = {compiled: CompiledGPUCommandGraph<void>; id: string};

export async function createPointCloudSupport(
  ctx: SceneContext<PointCloudSupportOptions>
): Promise<SceneInstance<PointCloudSupportOptions>> {
  const dataset = ctx.datasets.get('lidar-ground');
  const sourceCount = dataset.count;
  const sampleCount = Math.min(TARGET_ROWS, sourceCount);
  const projectedSource = dataset.projectColumn('position');
  const sourceElevations = dataset.column<Float32Array>('elevation');
  const sourceClassifications = dataset.column<Uint32Array>('classification');
  const positions = new Float32Array(sampleCount * 3);
  const positions2D = new Float32Array(sampleCount * 2);
  const classifications = new Uint32Array(sampleCount);
  const displayClasses = new Uint32Array(sampleCount);
  const sourceClassSlot = new Map([
    [1, 0],
    [2, 1],
    [7, 2],
    [9, 3],
    [18, 4],
    [20, 5]
  ]);
  const step = sourceCount / sampleCount;
  let minimumX = Number.POSITIVE_INFINITY;
  let minimumY = Number.POSITIVE_INFINITY;
  let minimumZ = Number.POSITIVE_INFINITY;
  let maximumX = Number.NEGATIVE_INFINITY;
  let maximumY = Number.NEGATIVE_INFINITY;
  let maximumZ = Number.NEGATIVE_INFINITY;
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const sourceIndex = Math.floor(sampleIndex * step);
    const x = projectedSource[sourceIndex * 2];
    const y = projectedSource[sourceIndex * 2 + 1];
    const z = sourceElevations[sourceIndex];
    positions[sampleIndex * 3] = x;
    positions[sampleIndex * 3 + 1] = y;
    positions[sampleIndex * 3 + 2] = z;
    positions2D[sampleIndex * 2] = x;
    positions2D[sampleIndex * 2 + 1] = y;
    classifications[sampleIndex] = sourceClassifications[sourceIndex];
    displayClasses[sampleIndex] = sourceClassSlot.get(sourceClassifications[sourceIndex]) ?? 6;
    minimumX = Math.min(minimumX, x);
    minimumY = Math.min(minimumY, y);
    minimumZ = Math.min(minimumZ, z);
    maximumX = Math.max(maximumX, x);
    maximumY = Math.max(maximumY, y);
    maximumZ = Math.max(maximumZ, z);
  }

  const gridCellSize = 2;
  const gridOrigin = [minimumX, minimumY] as const;
  const gridWidth = Math.ceil((maximumX - minimumX) / gridCellSize) + 1;
  const gridHeight = Math.ceil((maximumY - minimumY) / gridCellSize) + 1;
  const rasterGrid = {
    origin: gridOrigin,
    cellSize: gridCellSize,
    width: gridWidth,
    height: gridHeight
  };
  const cubeSize = Math.max(maximumX - minimumX, maximumY - minimumY, maximumZ - minimumZ) + 0.01;
  const octreeOrigin = [minimumX, minimumY, minimumZ] as const;
  const voxelOrigin = octreeOrigin;
  const voxelGrid = [
    Math.ceil((maximumX - minimumX) / VOXEL_SIZE_METERS) + 1,
    Math.ceil((maximumY - minimumY) / VOXEL_SIZE_METERS) + 1,
    Math.ceil((maximumZ - minimumZ) / VOXEL_SIZE_METERS) + 1
  ] as const;
  const voxelCount = voxelGrid.reduce((product, dimension) => product * dimension, 1);
  const partitionCount = Math.ceil(sampleCount / 2);
  const secondPartitionCount = sampleCount - partitionCount;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'point-cloud-support');
  const positionBuffer = resources.createBuffer('positions', positions);
  const position2DBuffer = resources.createBuffer('positions-2d', positions2D);
  const classificationBuffer = resources.createBuffer('source-classifications', classifications);
  const displayClassBuffer = resources.createBuffer('display-classes', displayClasses);
  const radiusCountsBuffer = resources.createBuffer('radius-counts', sampleCount * 4);
  const radiusMaskBuffer = resources.createBuffer('radius-mask', sampleCount * 4);
  const smrfSurfaceBuffer = resources.createBuffer('smrf-surface', gridWidth * gridHeight * 4);
  const smrfMaskBuffer = resources.createBuffer('smrf-ground-mask', sampleCount * 4);
  const smrfClassificationBuffer = resources.createBuffer('smrf-classifications', sampleCount * 4);
  const smrfStatusBuffer = resources.createBuffer('smrf-status', sampleCount * 4);
  const octreePointIndicesBuffer = resources.createBuffer('octree-point-indices', sampleCount * 4);
  const maximumOctreeDepth = 7;
  const octreeNodeCapacity = sampleCount * (maximumOctreeDepth + 1);
  const octreeNodesBuffer = resources.createBuffer('octree-nodes', octreeNodeCapacity * 8 * 4);
  const octreeStatusBuffer = resources.createBuffer('octree-status', 4 * 4);
  const octreeOwnershipBuffer = resources.createBuffer('octree-row-ownership', sampleCount * 4);
  const partBuffers = [0, 1].map(part => ({
    counts: resources.createBuffer(`voxel-part-${part}-counts`, voxelCount * 4),
    representatives: resources.createBuffer(`voxel-part-${part}-representatives`, voxelCount * 4),
    output: resources.createBuffer(`voxel-part-${part}-sample`, voxelCount * 12)
  }));
  const mergedCountsBuffer = resources.createBuffer('voxel-merged-counts', voxelCount * 4);
  const mergedCentroidsBuffer = resources.createBuffer('voxel-merged-centroids', voxelCount * 12);
  const mergedRepresentativesBuffer = resources.createBuffer(
    'voxel-merged-representatives',
    voxelCount * 4
  );
  const mergedStatusBuffer = resources.createBuffer('voxel-merged-status', voxelCount * 4);
  const directCountsBuffer = resources.createBuffer('voxel-direct-counts', voxelCount * 4);
  const directRepresentativesBuffer = resources.createBuffer(
    'voxel-direct-representatives',
    voxelCount * 4
  );
  const directOutputBuffer = resources.createBuffer('voxel-direct-output', voxelCount * 12);
  const voxelOnesBuffer = resources.createBuffer('voxel-ones', new Uint32Array(voxelCount).fill(1));
  const occupiedFlagsBuffer = resources.createBuffer('voxel-occupied-flags', voxelCount * 4);
  const compactVoxelIndicesBuffer = resources.createBuffer('voxel-compact-indices', voxelCount * 4);
  const compactVoxelPositionsBuffer = resources.createBuffer(
    'voxel-compact-positions',
    voxelCount * 8
  );
  const compactVoxelClassesBuffer = resources.createBuffer('voxel-compact-classes', voxelCount * 4);
  const octreeOwnershipClasses = new Uint32Array(sampleCount);
  const voxelDrawViewCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'point-cloud-support-voxel-view-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const voxelCommands = voxelDrawViewCommands;
  const voxelClassesBuffer = compactVoxelClassesBuffer;
  const voxelPositionsBuffer = compactVoxelPositionsBuffer;
  const compileGraph = async (settings: BuildSettings): Promise<AnalysisGraph> => {
    const graphId = `point-cloud-support-r${settings.radius}-k${settings.minimumNeighbors}-w${settings.smrfWindow}-s${settings.smrfSlope}-d${settings.maximumDepth}-n${settings.nodePointLimit}`;
    const graph = new GPUCommandGraph<void>(device, {id: graphId});
    const importView = <Format extends 'float32' | 'float32x2' | 'float32x3' | 'uint32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    const positionsView = importView('positions', positionBuffer, 'float32x3', sampleCount);
    const sourceClassView = importView('classes', classificationBuffer, 'uint32', sampleCount);
    const radiusCounts = importView('radius-counts', radiusCountsBuffer, 'uint32', sampleCount);
    const radiusMask = importView('radius-mask', radiusMaskBuffer, 'uint32', sampleCount);
    const smrfSurface = importView(
      'smrf-surface',
      smrfSurfaceBuffer,
      'float32',
      gridWidth * gridHeight
    );
    const smrfGroundMask = importView('smrf-mask', smrfMaskBuffer, 'uint32', sampleCount);
    const smrfClassifications = importView(
      'smrf-classes',
      smrfClassificationBuffer,
      'uint32',
      sampleCount
    );
    const smrfStatus = importView('smrf-status', smrfStatusBuffer, 'uint32', sampleCount);
    const octreePointIndices = importView(
      'octree-point-indices',
      octreePointIndicesBuffer,
      'uint32',
      sampleCount
    );
    const octreeNodes = importView(
      'octree-nodes',
      octreeNodesBuffer,
      'uint32',
      octreeNodeCapacity * 8
    );
    const octreeStatus = importView('octree-status', octreeStatusBuffer, 'uint32', 4);
    const voxelOnes = importView('voxel-ones', voxelOnesBuffer, 'uint32', voxelCount);
    const mergedCounts = importView(
      'voxel-merged-counts',
      mergedCountsBuffer,
      'uint32',
      voxelCount
    );
    const mergedCentroids = importView(
      'voxel-merged-centroids',
      mergedCentroidsBuffer,
      'float32x3',
      voxelCount
    );
    const mergedRepresentatives = importView(
      'voxel-merged-representatives',
      mergedRepresentativesBuffer,
      'uint32',
      voxelCount
    );
    const mergedStatus = importView(
      'voxel-merged-status',
      mergedStatusBuffer,
      'uint32',
      voxelCount
    );
    const directCounts = importView(
      'voxel-direct-counts',
      directCountsBuffer,
      'uint32',
      voxelCount
    );
    const directRepresentatives = importView(
      'voxel-direct-representatives',
      directRepresentativesBuffer,
      'uint32',
      voxelCount
    );
    const directOutput = importView(
      'voxel-direct-output',
      directOutputBuffer,
      'float32x3',
      voxelCount
    );
    const flags = importView('voxel-occupied-flags', occupiedFlagsBuffer, 'uint32', voxelCount);
    const compactVoxelIndices = importView(
      'voxel-compact-indices',
      compactVoxelIndicesBuffer,
      'uint32',
      voxelCount
    );
    const compactVoxelPositions = importView(
      'voxel-compact-positions',
      compactVoxelPositionsBuffer,
      'float32x2',
      voxelCount
    );
    const compactVoxelClasses = importView(
      'voxel-compact-classes',
      compactVoxelClassesBuffer,
      'uint32',
      voxelCount
    );
    const drawView = voxelCommands.importToGraph(graph);
    const drawCount = graph.createDataView(drawView.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    graph.add(
      new GPUPointCloudRadiusFilter({
        id: 'support-radius-filter',
        positions: positionsView,
        radius: settings.radius,
        minimumNeighbors: settings.minimumNeighbors,
        counts: radiusCounts,
        mask: radiusMask
      })
    );
    graph.add(
      new GPUPointCloudSMRF({
        id: 'support-smrf',
        positions: positionsView,
        classifications: sourceClassView,
        selection: radiusMask,
        grid: rasterGrid,
        window: settings.smrfWindow,
        slope: settings.smrfSlope,
        scalar: 1.25,
        threshold: 0.5,
        output: {
          groundSurface: smrfSurface,
          groundMask: smrfGroundMask,
          outputClassifications: smrfClassifications,
          status: smrfStatus
        }
      })
    );
    graph.add(
      new GPUPointCloudOctree({
        id: 'support-octree',
        positions: positionsView,
        origin: octreeOrigin,
        size: cubeSize,
        maximumDepth: settings.maximumDepth,
        nodePointLimit: settings.nodePointLimit,
        samplingLevels: 1,
        pointIndices: octreePointIndices,
        nodes: octreeNodes,
        status: octreeStatus
      })
    );
    const firstPartition = graph.createDataView(positionsView.buffer, {
      format: 'float32x3',
      length: partitionCount,
      byteOffset: 0
    });
    const secondPartition = graph.createDataView(positionsView.buffer, {
      format: 'float32x3',
      length: secondPartitionCount,
      byteOffset: partitionCount * 12
    });
    const mergeParts = [firstPartition, secondPartition].map((partPositions, partIndex) => {
      const part = partBuffers[partIndex];
      const partCounts = importView(
        `voxel-part-${partIndex}-counts`,
        part.counts,
        'uint32',
        voxelCount
      );
      const partRepresentatives = importView(
        `voxel-part-${partIndex}-representatives`,
        part.representatives,
        'uint32',
        voxelCount
      );
      const partOutput = importView(
        `voxel-part-${partIndex}-output`,
        part.output,
        'float32x3',
        voxelCount
      );
      return addPointCloudSupportVoxelPart(graph, {
        id: `support-voxel-part-${partIndex}`,
        positions: partPositions,
        sourcePositions: positions.subarray(
          partIndex === 0 ? 0 : partitionCount * 3,
          partIndex === 0 ? partitionCount * 3 : positions.length
        ),
        origin: voxelOrigin,
        voxelSize: VOXEL_SIZE_METERS,
        gridSize: voxelGrid,
        counts: partCounts,
        selectedPositions: partOutput,
        representatives: partRepresentatives,
        uploadMeans: means =>
          importView(
            `voxel-part-${partIndex}-means`,
            resources.createBuffer(`voxel-part-${partIndex}-means`, means),
            'float32x3',
            voxelCount
          ),
        sourceRowOffset: partIndex === 0 ? 0 : partitionCount
      });
    });
    graph.add(
      new GPUPointCloudVoxelMerge({
        id: 'support-voxel-merge',
        partitionPolicy: 'disjoint-owned-rows',
        parts: mergeParts,
        output: {
          counts: mergedCounts,
          centroids: mergedCentroids,
          representatives: mergedRepresentatives,
          status: mergedStatus
        }
      })
    );
    graph.add(
      new GPUPointCloudVoxelDownsample({
        id: 'support-voxel-direct-oracle',
        positions: positionsView,
        origin: voxelOrigin,
        voxelSize: VOXEL_SIZE_METERS,
        gridSize: voxelGrid,
        counts: directCounts,
        representatives: directRepresentatives,
        output: directOutput
      })
    );
    graph.add(
      new GPUElementwise({
        id: 'support-voxel-valid-flags',
        input: mergedCounts,
        inputB: voxelOnes,
        output: flags,
        operation: 'min'
      })
    );
    graph.add(
      new GPUCompaction({
        id: 'support-voxel-compact',
        input: mergedRepresentatives,
        flags,
        output: compactVoxelIndices,
        count: drawCount
      })
    );
    graph.add(
      new GPUGather({
        id: 'support-voxel-gather-position',
        source: importView('positions-2d', position2DBuffer, 'float32x2', sampleCount),
        indices: compactVoxelIndices,
        output: compactVoxelPositions
      })
    );
    graph.add(
      new GPUGather({
        id: 'support-voxel-gather-class',
        source: importView('display-classes', displayClassBuffer, 'uint32', sampleCount),
        indices: compactVoxelIndices,
        output: compactVoxelClasses
      })
    );

    const compiled = resources.track(await graph.compileAsync());
    return {compiled, id: graphId};
  };
  const graphSwap = resources.track(new GraphSwap());
  let activeGraph = await compileGraph(ctx.options);
  let octreeValid = false;
  let voxelMergeValid = false;
  let analysisDirty = true;

  const sourceViews = [
    {buffer: radiusCountsBuffer, size: sampleCount * 4},
    {buffer: radiusMaskBuffer, size: sampleCount * 4},
    {buffer: smrfMaskBuffer, size: sampleCount * 4},
    {buffer: smrfStatusBuffer, size: sampleCount * 4},
    {buffer: octreeStatusBuffer, size: 16},
    {buffer: octreeNodesBuffer, size: octreeNodeCapacity * 8 * 4},
    {buffer: octreePointIndicesBuffer, size: sampleCount * 4},
    {buffer: mergedCountsBuffer, size: voxelCount * 4},
    {buffer: mergedRepresentativesBuffer, size: voxelCount * 4},
    {buffer: mergedStatusBuffer, size: voxelCount * 4},
    {buffer: partBuffers[0].counts, size: voxelCount * 4},
    {buffer: partBuffers[1].counts, size: voxelCount * 4},
    {buffer: directCountsBuffer, size: voxelCount * 4},
    {buffer: directRepresentativesBuffer, size: voxelCount * 4}
  ];
  const summaryReader = new SummaryReader(
    resources,
    'point-cloud-support-summary',
    sourceViews,
    bytes => {
      let offset = 0;
      const readUint32 = (length: number): Uint32Array => {
        const view = new Uint32Array(bytes, offset, length);
        offset += length * 4;
        return view;
      };
      const gpuRadiusCounts = readUint32(sampleCount);
      const gpuRadiusMask = readUint32(sampleCount);
      const gpuGroundMask = readUint32(sampleCount);
      const gpuGroundStatus = readUint32(sampleCount);
      const octreeStatus = readUint32(4);
      const octreeNodes = readUint32(octreeNodeCapacity * 8);
      const octreePointIndices = readUint32(sampleCount);
      const mergedCounts = readUint32(voxelCount);
      const mergedRepresentatives = readUint32(voxelCount);
      const mergedStatus = readUint32(voxelCount);
      const firstPartCounts = readUint32(voxelCount);
      const secondPartCounts = readUint32(voxelCount);
      const directCounts = readUint32(voxelCount);
      const directRepresentatives = readUint32(voxelCount);
      const radiusBuckets = new Array<number>(7).fill(0);
      let radiusSupported = 0;
      for (let row = 0; row < sampleCount; row++) {
        const neighbors = gpuRadiusCounts[row];
        radiusBuckets[neighbors <= 4 ? neighbors : neighbors <= 8 ? 5 : 6]++;
        radiusSupported += gpuRadiusMask[row];
      }
      ctx.setReadout('radiusSupported', radiusSupported);
      const radiusLabels = ['0', '1', '2', '3', '4', '5–8', '9+'];
      ctx.setChart('radiusChart', {
        kind: 'bars',
        title: 'Other returns inside radius',
        labels: radiusLabels,
        values: radiusBuckets,
        color: 0,
        description: 'Exact all-pairs 3D neighbor counts read from the GPU.'
      } satisfies BarChartData);
      let groundRows = 0;
      let noTerrain = 0;
      let excluded = 0;
      let edgeRows = 0;
      let otherRows = 0;
      for (let row = 0; row < sampleCount; row++) {
        groundRows += gpuGroundMask[row];
        const status = gpuGroundStatus[row];
        if (status & 1 || status & 2) noTerrain++;
        else if (status & 4) excluded++;
        else if (status & 8) edgeRows++;
        else if (gpuGroundMask[row] === 0) otherRows++;
      }
      ctx.setReadout('smrfGround', groundRows);
      ctx.setReadout('smrfUnresolved', noTerrain + excluded);
      ctx.setChart('smrfChart', {
        kind: 'bars',
        title: 'SMRF output rows',
        labels: ['Ground mask', 'Other selected', 'No terrain', 'Excluded', 'Edge / inpainted'],
        values: [groundRows, otherRows, noTerrain, excluded, edgeRows],
        color: 2,
        description:
          'GPU status flags distinguish unsupported cells, excluded rows, and edge or inpainted surface estimates.'
      } satisfies BarChartData);

      octreeValid = octreeStatus[3] === 0;
      if (octreeValid) {
        const nodeCounts = new Array<number>(8).fill(0);
        const ownedCounts = new Array<number>(8).fill(0);
        octreeOwnershipClasses.fill(0);
        const nodeLength = Math.min(octreeStatus[0], octreeNodeCapacity);
        let ownershipRows = 0;
        for (let nodeIndex = 0; nodeIndex < nodeLength; nodeIndex++) {
          const nodeOffset = nodeIndex * 8;
          const level = octreeNodes[nodeOffset + 1];
          const pointOffset = octreeNodes[nodeOffset + 5];
          const pointLength = octreeNodes[nodeOffset + 6];
          if (level >= 8 || pointOffset + pointLength > octreeStatus[1]) continue;
          nodeCounts[level]++;
          ownedCounts[level] += pointLength;
          for (let item = pointOffset; item < pointOffset + pointLength; item++) {
            const row = octreePointIndices[item];
            if (row < sampleCount) octreeOwnershipClasses[row] = level;
          }
          ownershipRows += pointLength;
        }
        octreeValid = ownershipRows === octreeStatus[1] && octreeStatus[2] === 0;
        if (octreeValid) {
          octreeOwnershipBuffer.write(octreeOwnershipClasses);
          ctx.setReadout('octreeOwned', ownershipRows);
          ctx.setReadout(
            'octreeOverflow',
            octreeStatus[3] === 0 ? '0 · valid' : 'Overflow · rejected'
          );
          const labels = Array.from({length: 8}, (_, level) => [
            `L${level} nodes`,
            `L${level} owned`
          ]).flat();
          const values = nodeCounts.flatMap((nodes, level) => [nodes, ownedCounts[level]]);
          ctx.setChart('octreeChart', {
            kind: 'bars',
            title: 'Node and row ownership by level',
            labels,
            values,
            color: 1,
            description:
              'Every retained source row is owned by one node level; rows are not discarded to form the hierarchy.'
          } satisfies BarChartData);
        } else {
          ctx.setReadout('octreeOwned', 'Invalid ownership ranges');
          ctx.setReadout('octreeOverflow', 'Ownership mismatch · rejected');
          ctx.setChart('octreeChart', null);
        }
      } else {
        ctx.setReadout('octreeOwned', 'Overflow · result rejected');
        ctx.setReadout('octreeOverflow', 'Overflow · result rejected');
        ctx.setChart('octreeChart', null);
      }

      let mergedRows = 0;
      let directRows = 0;
      let partRows = 0;
      let representativeMismatchCount = 0;
      let mergeError = false;
      for (let cell = 0; cell < voxelCount; cell++) {
        const flags = mergedStatus[cell];
        if (flags !== 0) mergeError = true;
        mergedRows += mergedCounts[cell];
        directRows += directCounts[cell];
        partRows += firstPartCounts[cell] + secondPartCounts[cell];
        if (
          mergedCounts[cell] !== directCounts[cell] ||
          mergedCounts[cell] !== firstPartCounts[cell] + secondPartCounts[cell]
        )
          mergeError = true;
        if (mergedCounts[cell] > 0 && mergedRepresentatives[cell] !== directRepresentatives[cell]) {
          representativeMismatchCount++;
          mergeError = true;
        }
      }
      voxelMergeValid =
        !mergeError &&
        mergedRows === sampleCount &&
        directRows === sampleCount &&
        partRows === sampleCount;
      if (voxelMergeValid) {
        ctx.setReadout('mergeRows', mergedRows);
        ctx.setReadout(
          'mergeCountCheck',
          `${mergedRows} merged = ${directRows} direct · counts match`
        );
        ctx.setReadout('mergeRepresentativeMismatch', representativeMismatchCount);
        ctx.setReadout('mergeStatus', '0 · valid');
        ctx.setChart('mergeChart', {
          kind: 'bars',
          title: 'Independent GPU row totals',
          labels: ['Part A', 'Part B', 'Merged', 'Direct'],
          values: [
            firstPartCounts.reduce((sum, count) => sum + count, 0),
            secondPartCounts.reduce((sum, count) => sum + count, 0),
            mergedRows,
            directRows
          ],
          color: 3,
          description:
            'Per-part, merged and one-cloud GPU reductions all account for the sampled rows.'
        } satisfies BarChartData);
      } else {
        ctx.setReadout('mergeRows', 'Invalid · result rejected');
        ctx.setReadout('mergeCountCheck', 'Mismatch or overflow · rejected');
        ctx.setReadout('mergeRepresentativeMismatch', null);
        ctx.setReadout(
          'mergeStatus',
          mergeError ? 'Overflow / mismatch · rejected' : 'Incomplete row conservation · rejected'
        );
        ctx.setChart('mergeChart', null);
      }
      ctx.requestLayers();
    }
  );
  ctx.setReadout('sampleRows', sampleCount);
  ctx.setReadout('octreeOverflow', 'Waiting for GPU result');
  ctx.setReadout('mergeStatus', 'Waiting for GPU result');
  const origin = [dataset.defaultOrigin[0], dataset.defaultOrigin[1], 0] as [
    number,
    number,
    number
  ];
  const baseRadius = () => ctx.options.pointSize;
  const requestGraph = (settings: BuildSettings): void => {
    void graphSwap
      .request({
        build: () => compileGraph(settings),
        commit: next => {
          octreeValid = false;
          voxelMergeValid = false;
          const previous = activeGraph;
          activeGraph = next;
          resources.release(previous.compiled);
          analysisDirty = true;
          ctx.requestLayers();
        },
        discard: next => resources.release(next.compiled)
      })
      .catch(error => ctx.setStatus(`Graph rebuild failed: ${String(error)}`));
  };

  return {
    getCompiledGraphs: () => [activeGraph.compiled],
    setOption(id, _value, state) {
      if (
        [
          'radius',
          'minimumNeighbors',
          'smrfWindow',
          'smrfSlope',
          'maximumDepth',
          'nodePointLimit'
        ].includes(id)
      ) {
        requestGraph({
          radius: state.radius,
          minimumNeighbors: state.minimumNeighbors,
          smrfWindow: state.smrfWindow,
          smrfSlope: state.smrfSlope,
          maximumDepth: state.maximumDepth,
          nodePointLimit: state.nodePointLimit
        });
      } else ctx.requestLayers();
    },
    encode(commandEncoder) {
      if (analysisDirty) {
        activeGraph.compiled.encode(commandEncoder, {parameters: undefined});
        summaryReader.request(commandEncoder);
        analysisDirty = false;
      } else summaryReader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const base = {coordinateOrigin: origin, radiusPixels: baseRadius(), opacity: 0.9};
      if (ctx.options.view === 'radius')
        return [
          new SpatialAnalysisPointLayer({
            id: 'support-radius-map',
            positions: position2DBuffer,
            values: radiusMaskBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [206, 108, 96, 255],
              [116, 190, 159, 255]
            ],
            instanceCount: sampleCount,
            ...base
          })
        ];
      if (ctx.options.view === 'smrf')
        return [
          new SpatialAnalysisPointLayer({
            id: 'support-smrf-ground',
            positions: position2DBuffer,
            values: smrfMaskBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [206, 108, 96, 255],
              [116, 190, 159, 255]
            ],
            instanceCount: sampleCount,
            ...base
          }),
          new SpatialAnalysisPointLayer({
            id: 'support-smrf-status',
            positions: position2DBuffer,
            values: smrfStatusBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [239, 183, 91, 230],
            noDataColor: [0, 0, 0, 0],
            instanceCount: sampleCount,
            ...base,
            radiusPixels: baseRadius() + 1
          })
        ];
      if (ctx.options.view === 'octree') {
        if (!octreeValid) return [];
        return [
          new SpatialAnalysisPointLayer({
            id: 'support-octree-row-depth',
            positions: position2DBuffer,
            values: octreeOwnershipBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: OCTREE_COLORS,
            instanceCount: sampleCount,
            ...base
          })
        ];
      }
      if (ctx.options.view === 'merge') {
        if (!voxelMergeValid) return [];
        return [
          new SpatialAnalysisPointLayer({
            id: 'support-merged-voxel-classes',
            positions: voxelPositionsBuffer,
            values: voxelClassesBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: CLASS_COLORS,
            drawCommands: voxelCommands,
            ...base
          })
        ];
      }
      return [
        new SpatialAnalysisPointLayer({
          id: 'support-source-classes',
          positions: position2DBuffer,
          values: displayClassBuffer,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: CLASS_COLORS,
          instanceCount: sampleCount,
          ...base
        })
      ];
    },
    destroy() {
      summaryReader.stop();
      resources.destroy();
    }
  };
}
