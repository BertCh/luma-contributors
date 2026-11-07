// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMaxPRegions builds as many contiguous regions as possible while every region still holds at
// least `threshold` of an extensive attribute (population). A higher threshold gives fewer,
// larger regions.

import {
  getGPUMaxPRegionsParameterValues,
  GPU_MAX_P_STATUS,
  GPUMaxPRegions
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './max-p-regions.parameters';
import {uploadTractMesh} from './support/geometry';
import {createRegionScene, drawLabels} from './support/regions';

export default defineOperationRun({
  parameters,
  caption: 'The most regions that each reach the population threshold.',
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
          status: builder.allocate('status', 'uint32', GPU_MAX_P_STATUS.length).view
        }
      })
    );
    return {
      outputs: {labels, ...uploadTractMesh(builder, tracts)},
      update: values => threshold.write(Float32Array.of(values.threshold))
    };
  },
  rebuildOn: ['startCount', 'seed'],
  draw: ({inputs, outputs}) => drawLabels(inputs, outputs, outputs.labels)
});
