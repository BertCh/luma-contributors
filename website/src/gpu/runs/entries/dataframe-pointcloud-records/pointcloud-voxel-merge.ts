// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudVoxelMerge of two tiles' voxel reductions (GPUPointCloudVoxelDownsample on the same
// grid). The tiles own disjoint rows of one cloud; the merged voxel table is drawn as dots at the
// merged centroids, brighter where more points fell in the voxel.

import {GPUPointCloudVoxelMerge} from '@luma-contributors/analysis/pointcloud-analysis';
import {GPUPointCloudVoxelDownsample} from '@luma.gl/graph/gpu-pointcloud';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {createTerrainCloud, hashToUnit} from './_lib/helpers';
import {parameters} from './pointcloud-voxel-merge.parameters';

const POINT_COUNT = 6000;

export default defineOperationRun({
  parameters,
  caption: 'Merged voxel centroids from two tiles, by point count.',
  createInputs: ({random, world}) => createTerrainCloud(random, world.width, POINT_COUNT),
  build(builder, cloud, values) {
    const voxelSize = values.voxelSize;
    const gridSize = [
      Math.ceil(cloud.width / voxelSize),
      Math.ceil(1 / voxelSize),
      Math.ceil(0.5 / voxelSize)
    ] as const;
    const voxelCount = gridSize[0] * gridSize[1] * gridSize[2];
    // Split the cloud into two tiles that own disjoint rows.
    const secondShare = values.secondTileShare / 100;
    const tiles: number[][] = [[], []];
    for (let row = 0; row < POINT_COUNT; row++)
      tiles[hashToUnit(row) < secondShare ? 1 : 0].push(row);
    const parts = tiles.map((rows, tileIndex) => {
      const tilePositions = new Float32Array(rows.length * 3);
      rows.forEach((row, index) =>
        tilePositions.set(cloud.positions.subarray(row * 3, row * 3 + 3), index * 3)
      );
      const counts = builder.allocate(`tile-${tileIndex}-counts`, 'uint32', voxelCount);
      const centroids = builder.allocate(`tile-${tileIndex}-centroids`, 'float32x3', voxelCount);
      const representatives = builder.allocate(
        `tile-${tileIndex}-representatives`,
        'uint32',
        voxelCount
      );
      builder.graph.add(
        new GPUPointCloudVoxelDownsample({
          id: `tile-${tileIndex}`,
          positions: builder.upload(`tile-${tileIndex}-positions`, 'float32x3', tilePositions).view,
          origin: [0, 0, 0],
          voxelSize,
          gridSize,
          representatives: representatives.view,
          counts: counts.view,
          output: centroids.view
        })
      );
      return {
        counts: counts.view,
        centroids: centroids.view,
        representatives: representatives.view,
        sourceRowOffset: tileIndex === 0 ? 0 : tiles[0].length
      };
    });
    const mergedCounts = builder.allocate('merged-counts', 'uint32', voxelCount);
    const mergedCentroids = builder.allocate('merged-centroids', 'float32x3', voxelCount);
    builder.graph.add(
      new GPUPointCloudVoxelMerge({
        partitionPolicy: 'disjoint-owned-rows',
        parts,
        counts: mergedCounts.view,
        centroids: mergedCentroids.view,
        representatives: builder.allocate('merged-representatives', 'uint32', voxelCount).view,
        status: builder.allocate('status', 'uint32', voxelCount).view
      })
    );
    return {
      outputs: {
        footprint: builder.upload('footprint', 'float32x2', cloud.footprint),
        centers: addFloatCopy(builder, 'merged-footprint', mergedCentroids, voxelCount, {
          outputComponents: 2
        }),
        counts: addFloatCopy(builder, 'merged-counts-float', mergedCounts, voxelCount)
      }
    };
  },
  rebuildOn: ['voxelSize', 'secondTileShare'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.footprint,
      color: [0.5, 0.55, 0.65, 0.35],
      radius: 1.2
    },
    {
      layer: 'dots',
      positions: outputs.centers,
      values: outputs.counts,
      range: [0, 30],
      fade: true,
      radius: 4
    }
  ]
});
