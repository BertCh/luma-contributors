// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUWardRegions merges neighboring tracts bottom-up, always the pair that raises the within-
// region variance least, until `targetRegionCount` contiguous regions remain.

import {
  getGPUWardRegionsParameterValues,
  GPU_WARD_STATUS,
  GPUWardRegions
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './ward-regions.parameters';
import {uploadTractMesh} from './support/geometry';
import {createRegionScene, drawLabels} from './support/regions';

export default defineOperationRun({
  parameters,
  caption: 'Contiguous regions by Ward agglomeration.',
  createInputs: ({world}) => createRegionScene(makeRandom(7), world, 6, 2),
  build(builder, scene, current) {
    const {tracts, attributes, columnCount} = scene;
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    builder.graph.add(
      new GPUWardRegions({
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        values: builder.upload('attributes', 'float32', attributes).view,
        columnCount,
        standardize: current.standardize === 1,
        targetRegionCount: current.targetRegionCount,
        maximumIterations: tracts.count - current.targetRegionCount,
        mergeSchedule: current.mergeSchedule === 0 ? 'sequential' : 'mutual-nearest',
        parameters: builder.upload(
          'parameters',
          'uint32',
          getGPUWardRegionsParameterValues({seed: 1})
        ).view,
        output: {
          labels: labels.view,
          objective: builder.allocate('objective', 'float32', 1).view,
          status: builder.allocate('status', 'uint32', GPU_WARD_STATUS.length + 1).view
        }
      })
    );
    return {outputs: {labels, ...uploadTractMesh(builder, tracts)}};
  },
  draw: ({inputs, outputs}) => drawLabels(inputs, outputs, outputs.labels)
});
