// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialEmpiricalBayesRates: shrinks each tract's raw rate toward the rate of its neighbors
// instead of the overall rate, so local risk differences survive. Colors are the spatially
// smoothed rates. Hovering a tract outlines it and the neighbors it is shrunk toward, and reads its
// raw rate, its neighbors' mean raw rate and the smoothed rate.

import {GPUSpatialEmpiricalBayesRates} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {
  createTractOutline,
  getRateCounts,
  getTractNeighbors,
  pickTract,
  uploadTractMesh
} from './shared/tract-rates';
import {parameters} from './spatial-empirical-bayes-rates.parameters';

export default defineOperationRun({
  parameters,
  caption:
    "Age 65+ shares smoothed toward each tract's neighbors (Census counts with simulated sampling noise).",
  createInputs: ({random, world}) => ({tracts: createTracts(random, world), focus: {row: -1}}),
  build(builder, {tracts}) {
    const events = builder.allocate('events', 'float32', tracts.count);
    const populations = builder.allocate('populations', 'float32', tracts.count);
    const spatialRates = builder.allocate('spatial-rates', 'float32', tracts.count);
    const smoothedRates = builder.allocate('smoothed-rates', 'float32', tracts.count);
    const rawRates = builder.allocate('raw-rates', 'float32', tracts.count);
    builder.graph.add(
      new GPUSpatialEmpiricalBayesRates({
        events: events.view,
        populations: populations.view,
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        output: {spatialRates: spatialRates.view, smoothedRates: smoothedRates.view}
      })
    );
    return {
      outputs: {spatialRates, rawRates, ...uploadTractMesh(builder, tracts)},
      update: current => {
        const counts = getRateCounts(tracts, current.riskContrast, current.populationScale);
        events.write(counts.events);
        populations.write(counts.populations);
        rawRates.write(
          Float32Array.from(counts.events, (count, index) => count / counts.populations[index])
        );
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = pickTract(inputs.tracts, pointer, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const counts = getRateCounts(inputs.tracts, values.riskContrast, values.populationScale);
      const getRate = (tract: number) => counts.events[tract] / counts.populations[tract];
      const neighbors = getTractNeighbors(inputs.tracts, row);
      let neighborTotal = 0;
      for (const neighbor of neighbors) neighborTotal += getRate(neighbor);
      const smoothed = peek(outputs.spatialRates)?.[row];
      const percent = (rate: number) => `${(rate * 100).toFixed(0)}%`;
      return {
        glyphs: [
          createTractOutline(inputs.tracts, row),
          ...Array.from(neighbors, neighbor => createTractOutline(inputs.tracts, neighbor))
        ],
        readout: `raw ${percent(getRate(row))} · neighbors ${
          neighbors.length > 0 ? percent(neighborTotal / neighbors.length) : '-'
        } · smoothed ${smoothed === undefined ? '…' : percent(smoothed)}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.meshIndices,
      owners: outputs.owners,
      values: outputs.spatialRates,
      triangleCount: inputs.tracts.triangleCount,
      range: [0, 0.35]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount}
  ]
});
