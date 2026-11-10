// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNaturalNeighbor interpolates each grid cell with Sibson (Voronoi-area) weights of the
// surrounding samples; the surface is continuous and passes through every sample.

import {GPUNaturalNeighbor} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './natural-neighbor.parameters';
import {addChannel} from './support/channel';
import {createRunGrid} from './support/grid';
import {
  createHullSegments,
  createStretchedSamples,
  drawSampleDots,
  findNearestSamples
} from './support/interpolation';

export default defineOperationRun({
  parameters,
  caption:
    'Sibson weights from the Voronoi cells of the samples (dots); the surface stops at the hull of the samples (white). Move one sample.',
  createInputs: ({random, world}) => {
    const samples = createStretchedSamples(random, world, 60);
    // The sample nearest the middle follows the pointer; it moves to row 0 so every sample count keeps it.
    const [middle] = findNearestSamples(samples.positions, world.width / 2, 0.5, 1);
    swapRows(samples.positions, 2, 0, middle);
    swapRows(samples.values, 1, 0, middle);
    const carried = 0;
    return {
      grid: createRunGrid(world, 36),
      samples,
      carried,
      positions: Float32Array.from(samples.positions)
    };
  },
  build(builder, {grid, samples, positions}, values) {
    const cellCount = grid.columns * grid.rows;
    const sampleCount = values.sampleCount;
    const results = builder.allocate('results', 'float32x4', cellCount);
    const sampleBuffer = builder.upload(
      'positions',
      'float32x2',
      positions.slice(0, sampleCount * 2)
    );
    builder.graph.add(
      new GPUNaturalNeighbor({
        positions: sampleBuffer.view,
        values: builder.upload('values', 'float32', samples.values.slice(0, sampleCount)).view,
        queries: builder.upload('queries', 'float32x2', grid.centers).view,
        maximumVertexCount: values.maximumVertexCount,
        maximumNeighborCount: 64,
        output: {values: results.view, status: builder.allocate('status', 'uint32', cellCount).view}
      })
    );
    const hull = builder.upload(
      'hull',
      'float32x4',
      createHullSegments(positions, sampleCount, 64)
    );
    return {
      outputs: {
        hull,
        prediction: addChannel(builder, results, 0, 'prediction'),
        positions: sampleBuffer,
        sampleValues: builder.upload('sample-values', 'float32', samples.values),
        sampleCount
      },
      update: () => {
        sampleBuffer.write(positions.slice(0, sampleCount * 2));
        hull.write(createHullSegments(positions, sampleCount, 64));
      }
    };
  },
  rebuildOn: ['maximumVertexCount', 'sampleCount'],
  // The pointer carries one sample; the Voronoi cells and the surface re-form around it.
  interaction: {
    kind: 'drag',
    hint: 'carry a sample',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.samples.positions[inputs.carried * 2],
        inputs.samples.positions[inputs.carried * 2 + 1]
      ]),
    resolve(pointer, {inputs}) {
      const x = Math.min(inputs.grid.window[2], Math.max(0, pointer.x));
      const y = Math.min(inputs.grid.window[3], Math.max(0, pointer.y));
      inputs.positions[inputs.carried * 2] = x;
      inputs.positions[inputs.carried * 2 + 1] = y;
      return {
        rerun: true,
        glyphs: [{shape: 'ring', at: [x, y], radius: 0.035}],
        readout: `sample ${inputs.carried} = ${inputs.samples.values[inputs.carried].toFixed(2)} at ${x.toFixed(2)}, ${y.toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.prediction,
      columns: inputs.grid.columns,
      rows: inputs.grid.rows,
      range: [0, 1]
    },
    // The surface exists only inside the convex hull of the samples (white outline).
    {layer: 'lines', segments: outputs.hull, radius: 1.4, color: [1, 1, 1, 0.8]},
    ...drawSampleDots(outputs.positions, outputs.sampleValues, outputs.sampleCount)
  ]
});

function swapRows(array: Float32Array, width: number, first: number, second: number): void {
  for (let component = 0; component < width; component++) {
    const held = array[first * width + component];
    array[first * width + component] = array[second * width + component];
    array[second * width + component] = held;
  }
}
