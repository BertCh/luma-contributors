// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMaxPRegions builds as many contiguous regions as possible while every region still holds at
// least `threshold` of an extensive attribute (population). A higher threshold gives fewer,
// larger regions.

import {
  getGPUMaxPRegionsParameterValues,
  GPU_MAX_P_STATUS_LAYOUT,
  GPUMaxPRegions
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './max-p-regions.parameters';
import {uploadTractMesh} from './support/geometry';
import {bindAxes} from '../../interaction';
import {createRegionScene, drawLabels} from './support/regions';

const STRIP = [0.04, 0.03, 0.96, 0.2] as const;
// The strip's population axis; the highest threshold the slider offers.
const STRIP_MAXIMUM = 40000;

export default defineOperationRun({
  parameters,
  caption:
    "The most regions that each reach the population threshold; the bars are the regions' populations.",
  createInputs: ({world}) => createRegionScene(makeRandom(7), world, 8, 2),
  build(builder, scene, current) {
    const {tracts, attributes, columnCount, population} = scene;
    const threshold = builder.allocate('threshold', 'float32', 1);
    const parameterBuffer = builder.upload(
      'parameters',
      'uint32',
      getGPUMaxPRegionsParameterValues({seed: current.seed})
    );
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    builder.graph.add(
      new GPUMaxPRegions({
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        values: builder.upload('attributes', 'float32', attributes).view,
        columnCount,
        standardize: true,
        extensiveValues: builder.upload('population', 'float32', population).view,
        threshold: threshold.view,
        startCount: current.startCount,
        parameters: parameterBuffer.view,
        output: {
          labels: labels.view,
          objective: builder.allocate('objective', 'float32', 1).view,
          status: builder.allocate('status', 'uint32', GPU_MAX_P_STATUS_LAYOUT.length).view
        }
      })
    );
    return {
      outputs: {labels, ...uploadTractMesh(builder, tracts)},
      update: values => threshold.write(Float32Array.of(values.threshold))
    };
  },
  rebuildOn: ['startCount', 'seed'],
  // The population of every region, read back after each run, as bars under the map.
  async afterRun({outputs, inputs, upload}) {
    const bytes = await outputs.labels.buffer.readAsync();
    const labels = new Uint32Array(bytes.buffer, bytes.byteOffset, inputs.tracts.count);
    const sums = new Map<number, number>();
    labels.forEach((label, tract) => {
      if (label < inputs.tracts.count) {
        sums.set(label, (sums.get(label) ?? 0) + inputs.population[tract]);
      }
    });
    const bars = new Float32Array(inputs.tracts.count);
    [...sums.values()]
      .sort((first, second) => second - first)
      .forEach((sum, row) => (bars[row] = sum));
    return {bars: upload('region-population', 'float32', bars), regionCount: sums.size};
  },
  // x is the population threshold: the line across the bars is the least a region may hold.
  interaction: bindAxes(parameters, {
    kind: 'level',
    hint: 'set the threshold',
    x: 'threshold',
    decorate: (values, _pointer, {world}) => {
      const y = STRIP[1] + (values.threshold / STRIP_MAXIMUM) * (STRIP[3] - STRIP[1]);
      return {
        glyphs: [
          {shape: 'line', from: [STRIP[0] * world.width, y], to: [STRIP[2] * world.width, y]}
        ],
        readout: `threshold ${values.threshold.toLocaleString('en-US')} residents per region`
      };
    }
  }),
  draw: ({inputs, outputs, extra}) => [
    ...drawLabels(inputs, outputs, outputs.labels),
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.bars,
      range: [0, STRIP_MAXIMUM],
      rectangle: STRIP,
      ramp: 'sequential'
    }
  ]
});
