// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainPeakSnap on a Grand Canyon DEM: survey points that are close to, but not on, summits are
// moved to the highest cell within a search radius, unless the move or the change in height is too
// large. The dots are the snapped positions colored by the status the GPU wrote (snapped,
// unchanged, rejected); the grey dots are the candidates as supplied. A small kernel maps the
// index-space positions the operation wrote into the drawing world.

import {
  getGPUTerrainPeakSnapParameterValues,
  GPUTerrainPeakSnap
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {addPixelToWorld} from './support/pixel-to-world';
import {
  createTerrainInput,
  drawElevationContext,
  findSpreadLocalMaxima,
  getCellWorldPosition,
  getWorldPerMeter,
  uploadTerrain
} from './support/terrain';

import {parameters} from './peak-snap.parameters';

const CANDIDATE_COUNT = 11;

/** Index of the status codes in GPU_TERRAIN_PEAK_SNAP_STATUS: unchanged, snapped, onRing, ... */
const STATUS_NAMES = [
  'unchanged',
  'snapped',
  'on a ring',
  'move too long',
  'height off',
  'no data',
  'outside'
];
const STATUS_COLORS = [
  [0.55, 0.58, 0.65],
  [0.2, 0.8, 0.4],
  [0.95, 0.65, 0.2],
  [0.9, 0.25, 0.25],
  [0.65, 0.3, 0.85],
  [0.4, 0.4, 0.4],
  [0.4, 0.4, 0.4]
] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Elevation and rough peak positions in; positions moved onto the actual summits (or rejected) out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(world);
    const peaks = findSpreadLocalMaxima(terrain, CANDIDATE_COUNT, 8).map(index => ({
      column: index % terrain.columns,
      row: Math.floor(index / terrain.columns),
      // Surveyed height: the true summit height plus a survey error of up to 80 m.
      height: terrain.elevation[index] + (random() - 0.5) * 160,
      direction: random() * Math.PI * 2,
      amount: 0.4 + 0.6 * random()
    }));
    // The pointer carries one more candidate (column and row, fractional) beside the survey points.
    // Starts a few cells from the highest summit, so the first picture shows a snap.
    const first = peaks[0];
    const carried: [number, number] = [
      Math.min(first.column + 2, terrain.columns - 1),
      Math.min(first.row + 1, terrain.rows - 1)
    ];
    return {...terrain, peaks, carried};
  },
  build(builder, terrain) {
    const {grid, band, elevation} = uploadTerrain(builder, terrain);
    const count = terrain.peaks.length + 1;
    const candidates = builder.allocate('candidates', 'float32x2', count);
    const candidateHeights = builder.allocate('candidate-heights', 'float32', count);
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
      outputs: {elevation, snapped, heights, snappedDrawn, snapStatus, candidatesDrawn},
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
        const [carriedColumn, carriedRow] = terrain.carried;
        moved.push([carriedColumn, carriedRow]);
        candidates.write(Float32Array.from(moved.flat()));
        candidateHeights.write(
          Float32Array.from([
            ...terrain.peaks.map(peak => peak.height),
            terrain.elevation[Math.round(carriedRow) * terrain.columns + Math.round(carriedColumn)]
          ])
        );
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
  // The pointer carries one extra survey point: it snaps to the highest cell within the radius.
  interaction: {
    kind: 'drag',
    hint: 'drop a survey point',
    rest: ({inputs}) => [
      (inputs.carried[0] + 0.5) / inputs.columns,
      (inputs.carried[1] + 0.5) / inputs.rows
    ],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      inputs.carried = [
        Math.min(Math.max((pointer.x / world.width) * inputs.columns - 0.5, 0), inputs.columns - 1),
        Math.min(Math.max((1 - pointer.y / world.height) * inputs.rows - 0.5, 0), inputs.rows - 1)
      ];
      const worldPerMeter = getWorldPerMeter(inputs, world);
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: [pointer.x, pointer.y], radius: values.radius * worldPerMeter},
        {
          shape: 'ring',
          at: [pointer.x, pointer.y],
          radius: values.maximumMove * worldPerMeter,
          dashed: true
        }
      ];
      const snapped = peek(outputs.snapped);
      const statuses = peek(outputs.snapStatus);
      const heights = peek(outputs.heights);
      const carriedIndex = inputs.peaks.length;
      // Every survey point the operation moved shows its move, not only the carried one.
      const supplied = peek(outputs.candidatesDrawn);
      if (snapped && statuses && supplied) {
        inputs.peaks.forEach((_, index) => {
          const from = [supplied[index * 2], supplied[index * 2 + 1]] as const;
          // The ground each survey point may search: `radius` around it.
          glyphs.push({shape: 'ring', at: from, radius: values.radius * worldPerMeter});
          if (statuses[index] !== 1) return;
          const to = getCellWorldPosition(
            inputs,
            world,
            snapped[index * 2],
            snapped[index * 2 + 1]
          );
          glyphs.push({shape: 'arrow', from, to});
        });
      }
      let readout = `radius ${values.radius} m · maximumMove ${values.maximumMove} m`;
      if (snapped && statuses && heights) {
        // The arrow runs from the survey point to the cell it snapped to.
        if (statuses[carriedIndex] === 1) {
          glyphs.push({
            shape: 'arrow',
            from: [pointer.x, pointer.y],
            to: getCellWorldPosition(
              inputs,
              world,
              snapped[carriedIndex * 2],
              snapped[carriedIndex * 2 + 1]
            )
          });
        }
        readout = `${STATUS_NAMES[statuses[carriedIndex]] ?? 'unknown'} · top ${Math.round(heights[carriedIndex])} m · radius ${values.radius} m`;
      }
      return {rerun: true, glyphs, readout};
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    {layer: 'dots', context: true, positions: outputs.candidatesDrawn, radius: 5},
    {
      layer: 'dots',
      positions: outputs.snappedDrawn,
      values: outputs.snapStatus,
      ramp: 'categorical',
      categories: STATUS_COLORS,
      radius: 13
    }
  ]
});
