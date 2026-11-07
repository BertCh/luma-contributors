// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainPeakSnap on a synthetic DEM: survey points that are close to, but not on, summits are
// moved to the highest cell within a search radius, unless the move or the change in height is too
// large. The dots are the snapped positions colored by the status the GPU wrote (snapped,
// unchanged, rejected); the grey dots are the candidates as supplied. A small kernel maps the
// index-space positions the operation wrote into the drawing world.

import {
  getGPUTerrainPeakSnapParameterValues,
  GPUTerrainPeakSnap
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {addPixelToWorld} from './support/pixel-to-world';
import {
  createTerrainInput,
  drawElevationContext,
  findLocalMaxima,
  getCellWorldPosition,
  uploadTerrain
} from './support/terrain';
import {parameters} from './peak-snap.parameters';

const CANDIDATE_COUNT = 12;

export default defineOperationRun({
  parameters,
  caption:
    'Elevation and rough peak positions in; positions moved onto the actual summits (or rejected) out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(random, world);
    const peaks = findLocalMaxima(terrain, CANDIDATE_COUNT).map(index => ({
      column: index % terrain.columns,
      row: Math.floor(index / terrain.columns),
      // Surveyed height: the true summit height plus a survey error of up to 80 m.
      height: terrain.elevation[index] + (random() - 0.5) * 160,
      direction: random() * Math.PI * 2,
      amount: 0.4 + 0.6 * random()
    }));
    return {...terrain, peaks};
  },
  build(builder, terrain) {
    const {grid, band, elevation} = uploadTerrain(builder, terrain);
    const count = terrain.peaks.length;
    const candidates = builder.allocate('candidates', 'float32x2', count);
    const candidateHeights = builder.upload(
      'candidate-heights',
      'float32',
      Float32Array.from(terrain.peaks, peak => peak.height)
    );
    const candidatesDrawn = builder.allocate('candidates-drawn', 'float32x2', count);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const snapped = builder.allocate('snapped', 'float32x2', count);
    const heights = builder.allocate('heights', 'float32', count);
    const snapStatus = builder.allocate('snap-status', 'uint32', count);
    builder.graph.add(
      new GPUTerrainPeakSnap({
        grid,
        elevation: band,
        maximumRadiusPixels: 24,
        candidates: candidates.view,
        candidateHeights: candidateHeights.view,
        parameters: parameterBuffer.view,
        output: {positions: snapped.view, heights: heights.view, snapStatus: snapStatus.view}
      })
    );
    const snappedDrawn = addPixelToWorld(
      builder,
      'snapped-drawn',
      snapped,
      terrain,
      terrain.world,
      0.5
    );
    return {
      outputs: {elevation, snappedDrawn, snapStatus, candidatesDrawn},
      update: current => {
        const moved = terrain.peaks.map(peak => [
          Math.min(
            Math.max(peak.column + Math.cos(peak.direction) * peak.amount * current.offset, 0),
            terrain.columns - 1
          ),
          Math.min(
            Math.max(peak.row + Math.sin(peak.direction) * peak.amount * current.offset, 0),
            terrain.rows - 1
          )
        ]);
        candidates.write(Float32Array.from(moved.flat()));
        candidatesDrawn.write(
          Float32Array.from(
            moved.flatMap(([column, row]) =>
              getCellWorldPosition(terrain, terrain.world, column, row)
            )
          )
        );
        parameterBuffer.write(
          getGPUTerrainPeakSnapParameterValues(
            {
              radius: current.radius,
              maximumMove: current.maximumMove,
              maximumHeightChange: current.maximumHeightChange
            },
            grid
          )
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    {layer: 'dots', context: true, positions: outputs.candidatesDrawn, radius: 5},
    {layer: 'dots', positions: outputs.snappedDrawn, values: outputs.snapStatus, radius: 15}
  ]
});
