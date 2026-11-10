// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialLag over rook-contiguous tracts. The noisy tract values (dots) are replaced by the weighted sum of each tract's neighbors; with `normalize` that is the
// neighbor average, which smooths the map.

import {GPUSpatialLag} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {findTractUnder, getTractNeighbors, getTractOutline} from './shared/probes';
import {parameters} from './spatial-lag.parameters';

const SPIKE_VALUE = 3;

export default defineOperationRun({
  parameters,
  caption: 'Tract values replaced by the average of their neighbors.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    // The tract with a spike in its input value: the pointer moves it. Starts at the middle of the map.
    let middle = 0;
    for (let tract = 1; tract < tracts.count; tract++) {
      const distance = (index: number) =>
        Math.hypot(
          tracts.centers[index * 2] - world.width / 2,
          tracts.centers[index * 2 + 1] - 0.5
        );
      if (distance(tract) < distance(middle)) middle = tract;
    }
    return {...tracts, spike: {tract: middle}};
  },
  build(builder, tracts, values) {
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const noisy = builder.allocate('values', 'float32', tracts.count);
    const lag = builder.allocate('lag', 'float32', tracts.count);
    builder.graph.add(
      new GPUSpatialLag({
        values: noisy.view,
        weights,
        normalize: values.normalize === 1,
        output: lag.view
      })
    );
    return {
      outputs: {
        noisy,
        lag,
        vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
        indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
        owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
        centers: builder.upload('centers', 'float32x2', tracts.centers),
        outline: builder.upload('outline', 'float32x4', tracts.outline)
      },
      update: current => {
        const scaled = Float32Array.from(tracts.values, (value, index) => {
          const jitter = Math.sin(index * 12.9898) * 43758.5453;
          return value + (jitter - Math.floor(jitter) - 0.5) * current.noise * 0.6;
        });
        scaled[tracts.spike.tract] = SPIKE_VALUE;
        noisy.write(scaled);
      }
    };
  },
  rebuildOn: ['normalize'],
  // The pointer places a spike in one tract's input: the lag lights up its neighbors, not the tract itself.
  interaction: {
    kind: 'place',
    hint: 'place a spike',
    rest: state =>
      state.toUnit([
        state.inputs.centers[state.inputs.spike.tract * 2],
        state.inputs.centers[state.inputs.spike.tract * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, peek}) {
      const tract = findTractUnder(inputs, pointer.x, pointer.y);
      if (tract < 0) return {};
      const changed = tract !== inputs.spike.tract;
      inputs.spike.tract = tract;
      const lag = peek(outputs.lag);
      const neighbors = getTractNeighbors(inputs, tract);
      let neighborLag = 0;
      if (lag) for (const neighbor of neighbors) neighborLag += lag[neighbor] / neighbors.length;
      const glyphs: RunGlyph[] = [
        getTractOutline(inputs, tract),
        ...Array.from(neighbors, neighbor => getTractOutline(inputs, neighbor, 0, true))
      ];
      return {
        rerun: changed,
        focus: tract,
        glyphs,
        readout: `spike at tract ${tract} · lag in ${neighbors.length} neighbors ${neighborLag.toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.lag,
      triangleCount: inputs.triangleCount,
      range: [0, values.normalize ? 1.2 : 6]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.outlineCount
    },
    // The input values at the tract centers, on the same scale as the lag behind them.
    {
      layer: 'dots',
      positions: outputs.centers,
      values: outputs.noisy,
      range: [0, values.normalize ? 1.2 : 6],
      radius: 4
    }
  ]
});
