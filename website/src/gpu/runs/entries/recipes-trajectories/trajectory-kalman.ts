// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryKalman: noisy GPS fixes (dots) filtered forward (dim line) and smoothed with the
// RTS pass (bright line) over the true path (muted). The state buffers are [x, y, vx, vy] rows,
// so the line layer reads them as a node list whose even entries are the positions.

import {GPUTrajectoryKalman} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createFleet} from './lib/fleet';
import {parameters} from './trajectory-kalman.parameters';

/** Edges between consecutive observations of each track inside a float32x4 state buffer. */
function createStateEdges(trackOffsets: Uint32Array): Uint32Array {
  const edges: number[] = [];
  for (let track = 0; track < trackOffsets.length - 1; track++) {
    for (let row = trackOffsets[track]; row < trackOffsets[track + 1] - 1; row++) {
      edges.push(row * 2, (row + 1) * 2);
    }
  }
  return Uint32Array.from(edges);
}

export default defineOperationRun({
  parameters,
  caption: 'Noisy fixes, forward Kalman estimate and smoothed track.',
  createInputs: ({random, world}) => {
    const fleet = createFleet(world, {trackCount: 8, minimumFixes: 26, maximumFixes: 36});
    const unitNoise = Float32Array.from(
      {length: fleet.rowCount * 2},
      () => random() + random() - 1
    );
    return {...fleet, unitNoise};
  },
  build(builder, fleet) {
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const noisy = builder.allocate('noisy', 'float32x2', fleet.rowCount);
    const filteredState = builder.allocate('filtered', 'float32x4', fleet.rowCount);
    const filteredCovariance = builder.allocate('filtered-covariance', 'float32x4', fleet.rowCount);
    const smoothedState = builder.allocate('smoothed', 'float32x4', fleet.rowCount);
    const smoothedCovariance = builder.allocate('smoothed-covariance', 'float32x4', fleet.rowCount);
    const status = builder.allocate('status', 'uint32', fleet.rowCount);
    builder.graph.add(
      new GPUTrajectoryKalman({
        positions: noisy.view,
        times: builder.upload('times', 'float32', fleet.timestamps).view,
        offsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        parameters: parameterBuffer.view,
        output: {
          filteredState: filteredState.view,
          filteredCovariance: filteredCovariance.view,
          status: status.view,
          smoothedState: smoothedState.view,
          smoothedCovariance: smoothedCovariance.view
        }
      })
    );
    return {
      outputs: {
        noisy,
        filteredState,
        smoothedState,
        truth: builder.upload('truth', 'float32x4', fleet.segments),
        edges: builder.upload('edges', 'uint32x2', createStateEdges(fleet.trackOffsets))
      },
      update: current => {
        noisy.write(
          Float32Array.from(
            fleet.positions,
            (value, index) => value + fleet.unitNoise[index] * current.noise
          )
        );
        parameterBuffer.write(
          Float32Array.of(
            current.accelerationDensity * 1e-9,
            current.measurementVariance * 1e-6,
            1e-5,
            1000
          )
        );
      }
    };
  },
  // x trusts the motion model (accelerationDensity), y trusts the fixes (measurementVariance).
  interaction: bindAxes(parameters, {
    kind: 'pad',
    hint: 'model vs fixes',
    x: {parameter: 'accelerationDensity', scale: 'logarithmic'},
    y: {parameter: 'measurementVariance', scale: 'logarithmic'},
    decorate: (values, pointer) => ({
      glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
      readout: `q ${values.accelerationDensity}e-9 · r ${values.measurementVariance}e-6`
    })
  }),
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.truth, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.noisy, radius: 3.5},
    {
      layer: 'segments',
      nodes: outputs.filteredState,
      edges: outputs.edges,
      color: [0.45, 0.75, 1, 0.45],
      radius: 3.5
    },
    {
      layer: 'segments',
      nodes: outputs.smoothedState,
      edges: outputs.edges,
      color: [1, 0.78, 0.25, 1],
      radius: 1.8
    }
  ]
});
