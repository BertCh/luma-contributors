// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULagrangianFlowMap: every seed is carried through the Helene wind for `stepCount` RK2 steps of
// `timeStep` and its final position is written. A coarse grid of seeds shows the map as
// displacement lines (start to end); the reader places a cluster of seeds and sees where it ends up.
// A negative `timeStep` integrates backward: where the air came from. Steady field (one snapshot).

import {
  getGPULagrangianFlowMapParameterValues,
  GPULagrangianFlowMap
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {addDisplacementSegments} from './_lib/kernels';
import {createWind} from './_lib/wind';
import {parameters} from './lagrangian-flow-map.parameters';

const STEP_COUNTS = [16, 48, 96];
const GRID_ROWS = 14;
const CLUSTER_SIZE = 256;
const CLUSTER_RADIUS = 0.06;
const REST = [0.35, 0.5] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['stepCount'],
  caption:
    'Seeds carried through the Helene wind (steady: one snapshot); lines join start to end. Place a ring of seeds and see the patch it becomes. Time units are uncalibrated (1 m/s = 0.01 field heights per unit).',
  createInputs: ({aspect}) => {
    const field = createWind(aspect);
    const width = field.columns * field.cellSize;
    const gridColumns = Math.round(GRID_ROWS * width);
    const gridSeeds = Float32Array.from({length: gridColumns * GRID_ROWS * 2}, (_, item) => {
      const seed = item >> 1;
      return item & 1
        ? (Math.floor(seed / gridColumns) + 0.5) / GRID_ROWS
        : (((seed % gridColumns) + 0.5) * width) / gridColumns;
    });
    // A sunflower disc of offsets around the release point.
    const offsets = Float32Array.from({length: CLUSTER_SIZE * 2}, (_, item) => {
      const seed = item >> 1;
      const radius = CLUSTER_RADIUS * Math.sqrt((seed + 0.5) / CLUSTER_SIZE);
      const angle = seed * 2.399963;
      return radius * (item & 1 ? Math.sin(angle) : Math.cos(angle));
    });
    return {
      ...field,
      width,
      gridSeeds,
      gridCount: gridColumns * GRID_ROWS,
      offsets,
      release: [REST[0] * width, 1 - REST[1]] as [number, number]
    };
  },
  build(builder, field, values) {
    const stepCount = STEP_COUNTS[Math.round(values.stepCount)];
    const velocities = builder.upload('velocities', 'float32x2', field.velocities);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const gridSeeds = builder.upload('grid-seeds', 'float32x2', field.gridSeeds);
    const gridEnds = builder.allocate('grid-ends', 'float32x2', field.gridCount);
    const clusterSeeds = builder.allocate('cluster-seeds', 'float32x2', CLUSTER_SIZE);
    const clusterEnds = builder.allocate('cluster-ends', 'float32x2', CLUSTER_SIZE);
    for (const [id, seeds, ends] of [
      ['grid-flow-map', gridSeeds, gridEnds],
      ['cluster-flow-map', clusterSeeds, clusterEnds]
    ] as const) {
      builder.graph.add(
        new GPULagrangianFlowMap({
          id,
          seeds: seeds.view,
          velocitiesStart: velocities.view,
          velocitiesEnd: velocities.view,
          fieldWidth: field.columns,
          fieldHeight: field.rows,
          stepCount,
          parameters: parameterBuffer.view,
          output: {positions: ends.view}
        })
      );
    }
    const segments = builder.allocate('segments', 'float32x4', field.gridCount);
    addDisplacementSegments(builder, 'grid-segments', gridSeeds, gridEnds, segments);
    return {
      outputs: {
        speeds: builder.upload('speeds', 'float32', field.speeds),
        segments,
        gridEnds,
        clusterSeeds,
        clusterEnds
      },
      update: current => {
        parameterBuffer.write(
          getGPULagrangianFlowMapParameterValues({
            fieldExtent: [0, 0, field.cellSize, field.cellSize],
            startTime: 0,
            timeStep: current.timeStep,
            sliceStartTime: 0,
            sliceEndTime: 1
          })
        );
        clusterSeeds.write(field.offsets.map((offset, item) => offset + field.release[item & 1]));
      }
    };
  },
  interaction: {
    kind: 'place',
    hint: 'place the seeds',
    rest: REST,
    resolve(pointer, {inputs, outputs, values, peek}) {
      const moved = pointer.x !== inputs.release[0] || pointer.y !== inputs.release[1];
      inputs.release = [pointer.x, pointer.y];
      const stepCount = STEP_COUNTS[Math.round(values.stepCount)];
      const time = stepCount * values.timeStep;
      const spread = getPatchSpread(peek(outputs.clusterEnds), inputs.release);
      return {
        rerun: moved,
        glyphs: [{shape: 'ring', at: inputs.release, radius: CLUSTER_RADIUS}],
        readout: `moved ${(100 * spread.moved).toFixed(0)}% of field height · stretched ${spread.stretch.toFixed(1)}x · ${spread.lost} lost · T ${time > 0 ? '+' : ''}${time.toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.speeds,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      smoothing: true
    },
    {layer: 'lines', segments: outputs.segments, color: [1, 1, 1, 0.75]},
    {layer: 'dots', positions: outputs.gridEnds, radius: 2, color: [1, 1, 1, 0.9]},
    {layer: 'dots', context: true, positions: outputs.clusterSeeds, radius: 1.5},
    {layer: 'dots', positions: outputs.clusterEnds, radius: 2.5, color: [1, 0.55, 0.2, 1]}
  ]
});

/** How far the placed patch moved (centroid) and how much it stretched (longest extent / ring diameter). */
function getPatchSpread(ends: ArrayLike<number> | null | undefined, release: readonly number[]) {
  if (!ends) return {moved: 0, stretch: 1, lost: 0};
  let lost = 0;
  let sumX = 0;
  let sumY = 0;
  for (let seed = 0; seed < CLUSTER_SIZE; seed++) {
    const x = ends[seed * 2];
    const y = ends[seed * 2 + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      lost++;
      continue;
    }
    sumX += x;
    sumY += y;
  }
  const kept = CLUSTER_SIZE - lost;
  if (kept === 0) return {moved: 0, stretch: 0, lost};
  const centerX = sumX / kept;
  const centerY = sumY / kept;
  // Longest extent: the largest projection spread over a few directions.
  let longest = 0;
  for (let angle = 0; angle < Math.PI; angle += Math.PI / 12) {
    const directionX = Math.cos(angle);
    const directionY = Math.sin(angle);
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    for (let seed = 0; seed < CLUSTER_SIZE; seed++) {
      const x = ends[seed * 2];
      const y = ends[seed * 2 + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const projection = x * directionX + y * directionY;
      minimum = Math.min(minimum, projection);
      maximum = Math.max(maximum, projection);
    }
    longest = Math.max(longest, maximum - minimum);
  }
  return {
    moved: Math.hypot(centerX - release[0], centerY - release[1]),
    stretch: longest / (2 * CLUSTER_RADIUS),
    lost
  };
}
