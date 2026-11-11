// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared by the two home-range runs: one San Francisco Bay vessel track, a grid over the frame and
// the Brownian bridge graph. The y axis is mirrored for the operation (its grid row 0 is the
// smallest y, the raster layer draws row 0 at the top); a mirror keeps distances, so the density is
// the same. A CPU copy of the density formula places the pointer's rest position.

import {
  getGPUBrownianBridgeUtilizationParameterValues,
  GPUBrownianBridgeUtilization
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import type {FleetInput} from '../../recipes-trajectories/lib/fleet';
import {createMovementFleet} from './movement';

export const GRID_ROWS = 64;
export const SUB_STEP_COUNT = 15;

export type HomeRangeInputs = FleetInput & {
  metersPerUnit: number;
  world: {width: number; height: number};
  columns: number;
  rows: number;
  /** Positions with y mirrored (1 - y), as the operation reads them. */
  mirrored: Float32Array;
};

export function createHomeRangeInputs(world: {width: number; height: number}): HomeRangeInputs {
  const all = createMovementFleet(world, 8);
  // The track that spends the most distance under way (moored vessels give a few saturated blobs).
  let best = 0;
  let bestLength = -1;
  for (let track = 0; track < all.trackCount; track++) {
    let length = 0;
    for (let row = all.trackOffsets[track]; row < all.trackOffsets[track + 1] - 1; row++) {
      length += Math.hypot(
        all.positions[row * 2 + 2] - all.positions[row * 2],
        all.positions[row * 2 + 3] - all.positions[row * 2 + 1]
      );
    }
    if (length > bestLength) {
      best = track;
      bestLength = length;
    }
  }
  const first = all.trackOffsets[best];
  const last = all.trackOffsets[best + 1];
  const positions = all.positions.slice(first * 2, last * 2);
  const fleet = {
    ...all,
    trackCount: 1,
    rowCount: last - first,
    positions,
    timestamps: all.timestamps.slice(first, last),
    trackOffsets: Uint32Array.of(0, last - first),
    segments: all.segments.slice((first - best) * 4, (last - best - 1) * 4),
    segmentCount: last - first - 1
  };
  const mirrored = Float32Array.from(positions, (value, index) =>
    index % 2 === 1 ? 1 - value : value
  );
  return {
    ...fleet,
    world,
    columns: Math.round(GRID_ROWS * world.width),
    rows: GRID_ROWS,
    mirrored
  };
}

export type BridgeBuffers = {
  density: RunBuffer<'float32'>;
  segments: RunBuffer<'float32x4'>;
  write(motionVariance: number, locationError: number): void;
};

/** Adds GPUBrownianBridgeUtilization over the whole frame. */
export function addBrownianBridge(builder: RunBuilder, inputs: HomeRangeInputs): BridgeBuffers {
  const motionVariances = builder.allocate('motion-variances', 'float32', inputs.rowCount);
  const locationErrors = builder.allocate('location-errors', 'float32', inputs.rowCount);
  const parameterBuffer = builder.allocate('bridge-parameters', 'float32', 4);
  const density = builder.allocate('density', 'float32', inputs.columns * inputs.rows);
  builder.graph.add(
    new GPUBrownianBridgeUtilization({
      positions: builder.upload('positions', 'float32x2', inputs.mirrored).view,
      timestamps: builder.upload('timestamps', 'float32', inputs.timestamps).view,
      trackOffsets: builder.upload('offsets', 'uint32', inputs.trackOffsets).view,
      motionVariances: motionVariances.view,
      locationErrors: locationErrors.view,
      gridSize: [inputs.columns, inputs.rows],
      parameters: parameterBuffer.view,
      subStepCount: SUB_STEP_COUNT,
      output: {density: density.view}
    })
  );
  return {
    density,
    segments: builder.upload('segments', 'float32x4', inputs.segments),
    write(motionVariance, locationError) {
      parameterBuffer.write(
        getGPUBrownianBridgeUtilizationParameterValues({
          extent: [0, 0, inputs.world.width, inputs.world.height]
        })
      );
      motionVariances.write(new Float32Array(inputs.rowCount).fill(motionVariance));
      locationErrors.write(new Float32Array(inputs.rowCount).fill(locationError));
    }
  };
}

/** CPU density of the same formula (midpoint rule), row-major with row 0 at the top. */
export function getReferenceDensity(
  inputs: HomeRangeInputs,
  motionVariance: number,
  locationError: number
): Float32Array {
  const {columns, rows, positions, timestamps, trackOffsets, world} = inputs;
  const density = new Float64Array(columns * rows);
  let totalTime = 0;
  const errorVariance = locationError * locationError;
  for (let track = 0; track < trackOffsets.length - 1; track++) {
    for (let fix = trackOffsets[track]; fix < trackOffsets[track + 1] - 1; fix++) {
      const duration = timestamps[fix + 1] - timestamps[fix];
      if (duration <= 0) continue;
      totalTime += duration;
      for (let step = 0; step < SUB_STEP_COUNT; step++) {
        const alpha = (step + 0.5) / SUB_STEP_COUNT;
        const variance =
          duration * alpha * (1 - alpha) * motionVariance +
          ((1 - alpha) ** 2 + alpha ** 2) * errorVariance;
        if (variance <= 0) continue;
        const centerX = positions[fix * 2] + alpha * (positions[fix * 2 + 2] - positions[fix * 2]);
        const centerY =
          positions[fix * 2 + 1] + alpha * (positions[fix * 2 + 3] - positions[fix * 2 + 1]);
        for (let row = 0; row < rows; row++) {
          const y = 1 - (row + 0.5) / rows;
          for (let column = 0; column < columns; column++) {
            const x = ((column + 0.5) / columns) * world.width;
            const squared = (x - centerX) ** 2 + (y - centerY) ** 2;
            if (squared > 49 * variance) continue;
            density[row * columns + column] +=
              ((duration / SUB_STEP_COUNT) * Math.exp(-squared / (2 * variance))) /
              (2 * Math.PI * variance);
          }
        }
      }
    }
  }
  return Float32Array.from(density, value => value / totalTime);
}

/** Cumulative share of the total mass held by cells at least as dense as each cell. */
export function getReferenceVolume(density: Float32Array): Float32Array {
  const order = Array.from(density.keys()).sort(
    (first, second) => density[second] - density[first]
  );
  let total = 0;
  for (const value of density) total += value;
  const volume = new Float32Array(density.length);
  let running = 0;
  for (const cell of order) {
    running += density[cell];
    volume[cell] = running / total;
  }
  return volume;
}
