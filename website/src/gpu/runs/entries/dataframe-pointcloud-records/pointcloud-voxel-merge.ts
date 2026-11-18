// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudVoxelMerge of two tiles' voxel reductions (GPUPointCloudVoxelDownsample on the same
// grid). The tiles own disjoint rows of one cloud (points coloured by tile, the seam a line); the
// merged voxel table is drawn as dots at the merged centroids, brighter where more points fell in
// the voxel, over the voxel grid. A voxel that both tiles reduced is outlined: those are the cells
// the merge combined.

import {GPUPointCloudVoxelMerge} from '@luma-contributors/analysis/pointcloud-analysis';
import {GPUPointCloudVoxelDownsample} from '@luma.gl/graph/gpu-pointcloud';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {createLegend, mergeText, type RunLabel} from '../../text';
import {addFloatCopy} from './_lib/convert';
import {computePointCloudVoxelMeans} from './_lib/pointcloud-voxel-means';
import {createTerrainCloud} from './_lib/helpers';
import {parameters} from './pointcloud-voxel-merge.parameters';

const POINT_COUNT = 6000;
const SHARED_LIMIT = 200;
// Rows a tile does not own are moved here, outside every voxel grid.
const OUTSIDE = -10;
const TILE_COLORS = [
  [0.3, 0.7, 1],
  [1, 0.6, 0.2]
] as const;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  return {...createTerrainCloud(random, world.width, POINT_COUNT), world};
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  footprint: RunBuffer;
  tileOf: RunBuffer;
  centers: RunBuffer;
  counts: RunBuffer;
  firstCounts: RunBuffer;
  secondCounts: RunBuffer;
  shared: RunBuffer;
  sharedBoxes: RunBuffer;
  gridLines: RunBuffer;
  seam: RunBuffer;
};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Merged voxel centroids from two tiles split at the seam (left tile blue, right tile orange); outlined boxes mark voxels both tiles contributed to, and the larger dots are the merged voxel centroids.',
  createInputs,
  build(builder, cloud, values) {
    const voxelSize = values['voxelSize'];
    let highestPoint = 0;
    for (let index = 2; index < cloud.positions.length; index += 3)
      highestPoint = Math.max(highestPoint, cloud.positions[index]);
    const gridSize = [
      Math.ceil(cloud.width / voxelSize),
      Math.ceil(1 / voxelSize),
      Math.floor(highestPoint / voxelSize) + 1
    ] as const;
    const voxelCount = gridSize[0] * gridSize[1] * gridSize[2];
    // Each tile reads the whole cloud, but only owns the rows on its side of the seam: the other
    // rows are moved outside the grid, so the two tiles own disjoint rows.
    const tilePositions = [0, 1].map(tileIndex =>
      builder.allocate(`tile-${tileIndex}-positions`, 'float32x3', POINT_COUNT)
    );
    const tileCounts: RunBuffer[] = [];
    const tileMeans: RunBuffer[] = [];
    const parts = [0, 1].map(tileIndex => {
      const counts = builder.allocate(`tile-${tileIndex}-counts`, 'uint32', voxelCount);
      tileCounts.push(counts);
      const centroids = builder.allocate(`tile-${tileIndex}-centroids`, 'float32x3', voxelCount);
      tileMeans.push(centroids);
      const representatives = builder.allocate(
        `tile-${tileIndex}-representatives`,
        'uint32',
        voxelCount
      );
      builder.graph.add(
        new GPUPointCloudVoxelDownsample({
          id: `tile-${tileIndex}`,
          positions: tilePositions[tileIndex].view,
          origin: [0, 0, 0],
          voxelSize,
          gridSize,
          representatives: representatives.view,
          counts: counts.view,
          output: builder.allocate(`tile-${tileIndex}-selected-positions`, 'float32x3', voxelCount)
            .view
        })
      );
      return {
        counts: counts.view,
        centroids: centroids.view,
        representatives: representatives.view,
        sourceRowOffset: 0
      };
    });
    const mergedCounts = builder.allocate('merged-counts', 'uint32', voxelCount);
    const mergedCentroids = builder.allocate('merged-centroids', 'float32x3', voxelCount);
    builder.graph.add(
      new GPUPointCloudVoxelMerge({
        partitionPolicy: 'disjoint-owned-rows',
        parts,
        output: {
          counts: mergedCounts.view,
          centroids: mergedCentroids.view,
          representatives: builder.allocate('merged-representatives', 'uint32', voxelCount).view,
          status: builder.allocate('status', 'uint32', voxelCount).view
        }
      })
    );
    const tileOf = builder.allocate('tile-of', 'uint32', POINT_COUNT);
    // The voxel grid over the footprint, as line segments.
    const lines: number[] = [];
    for (let column = 0; column <= gridSize[0]; column++) {
      lines.push(column * voxelSize, 0, column * voxelSize, gridSize[1] * voxelSize);
    }
    for (let row = 0; row <= gridSize[1]; row++) {
      lines.push(0, row * voxelSize, gridSize[0] * voxelSize, row * voxelSize);
    }
    const seam = builder.allocate('seam', 'float32x4', 1);
    return {
      outputs: {
        footprint: builder.upload('footprint', 'float32x2', cloud.footprint),
        tileOf,
        centers: addFloatCopy(builder, 'merged-footprint', mergedCentroids, voxelCount, {
          outputComponents: 2
        }),
        counts: addFloatCopy(builder, 'merged-counts-float', mergedCounts, voxelCount),
        firstCounts: tileCounts[0],
        secondCounts: tileCounts[1],
        shared: builder.allocate('shared', 'float32x2', SHARED_LIMIT),
        sharedBoxes: builder.allocate('shared-boxes', 'float32x4', SHARED_LIMIT * 4),
        gridLines: builder.upload('grid-lines', 'float32x4', Float32Array.from(lines)),
        seam
      },
      update: current => {
        const owners = Uint32Array.from({length: POINT_COUNT}, (_, row) =>
          Number(cloud.positions[row * 3] >= current['tileBoundary'])
        );
        tileOf.write(owners);
        seam.write(Float32Array.of(current['tileBoundary'], 0, current['tileBoundary'], 1));
        for (const tileIndex of [0, 1]) {
          const ownedPositions = Float32Array.from({length: POINT_COUNT * 3}, (_, index) => {
            const row = Math.floor(index / 3);
            return owners[row] === tileIndex ? cloud.positions[index] : OUTSIDE;
          });
          tilePositions[tileIndex].write(ownedPositions);
          tileMeans[tileIndex].write(
            computePointCloudVoxelMeans(ownedPositions, voxelSize, gridSize)
          );
        }
      }
    };
  },
  rebuildOn: ['voxelSize'],
  // The pointer is the seam between the two tiles: x is where tile 0 ends and tile 1 begins. The
  // voxels that both tiles reduced (and the merge combined) are ringed.
  interaction: {
    kind: 'drag',
    hint: 'drag the seam',
    rest: ({world}) => [findParameter(parameters, 'tileBoundary').default / world.width, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const boundary = snapToParameter(findParameter(parameters, 'tileBoundary'), pointer.x);
      let first = 0;
      for (let row = 0; row < POINT_COUNT; row++) {
        if (inputs.positions[row * 3] < boundary) first++;
      }
      const firstCounts = peek(outputs.firstCounts);
      const secondCounts = peek(outputs.secondCounts);
      const centers = peek(outputs.centers);
      const rings = new Float32Array(SHARED_LIMIT * 2).fill(OUTSIDE);
      const boxes = new Float32Array(SHARED_LIMIT * 16).fill(OUTSIDE);
      const size = values['voxelSize'];
      let shared = 0;
      for (
        let voxel = 0;
        firstCounts && secondCounts && centers && voxel < firstCounts.length;
        voxel++
      ) {
        if (firstCounts[voxel] > 0 && secondCounts[voxel] > 0 && shared < SHARED_LIMIT) {
          rings.set([centers[voxel * 2], centers[voxel * 2 + 1]], shared * 2);
          // The outline of the voxel's cell in the footprint, as four segments.
          const left = Math.floor(centers[voxel * 2] / size) * size;
          const bottom = Math.floor(centers[voxel * 2 + 1] / size) * size;
          boxes.set(
            [
              left,
              bottom,
              left + size,
              bottom,
              left + size,
              bottom,
              left + size,
              bottom + size,
              left + size,
              bottom + size,
              left,
              bottom + size,
              left,
              bottom + size,
              left,
              bottom
            ],
            shared * 16
          );
          shared++;
        }
      }
      outputs.shared.write(rings);
      outputs.sharedBoxes.write(boxes);
      return {
        values: {tileBoundary: boundary},
        focus: shared,
        glyphs: [{shape: 'curtain', x: boundary}],
        readout: `voxelSize ${values['voxelSize']} · tiles ${first} | ${POINT_COUNT - first} pts · ${
          firstCounts ? shared : '…'
        } voxels shared`
      };
    }
  },
  draw: ({outputs, interaction, values, theme}) => {
    const sharedCount = typeof interaction.focus === 'number' ? interaction.focus : 0;
    const labels: RunLabel[] = [
      {
        text: 'seam',
        at: [values['tileBoundary'], 0.97],
        anchor: 'top-left',
        offset: [4, 0],
        role: 'title'
      }
    ];
    return [
      {layer: 'lines', context: true, segments: outputs.gridLines, radius: 0.8},
      {
        layer: 'dots',
        positions: outputs.footprint,
        values: outputs.tileOf,
        ramp: 'categorical',
        categories: TILE_COLORS,
        alpha: 0.9,
        radius: 1.8
      },
      {
        layer: 'dots',
        positions: outputs.centers,
        values: outputs.counts,
        range: [0, 6],
        fade: true,
        radius: 3
      },
      {layer: 'lines', segments: outputs.seam, color: [...theme.ink, 0.9], radius: 1.4},
      {
        layer: 'lines',
        segments: outputs.sharedBoxes,
        count: sharedCount * 4,
        color: [...theme.ink, 1],
        radius: 1.4
      },
      {
        layer: 'text',
        ...mergeText(
          {labels},
          createLegend({
            corner: 'bottom-left',
            entries: [
              {label: 'points of tile 1 (left of the seam)', swatch: TILE_COLORS[0]},
              {label: 'points of tile 2 (right of the seam)', swatch: TILE_COLORS[1]},
              {
                label: 'merged voxel centroid (output; more points, stronger)',
                swatch: [...theme.ramp[4]]
              },
              {label: `voxels both tiles reduced: ${sharedCount} (outlined)`, swatch: theme.ink}
            ]
          })
        )
      }
    ];
  }
});
