// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSkaterRegions cuts the minimum spanning tree (from GPUSpatialWeightsMinimumSpanningTree) at
// its most dissimilar edges into contiguous regions. Colors are the region `labels` output.

import {
  getGPUSkaterRegionsParameterValues,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './skater-regions.parameters';
import {uploadTractMesh} from './support/geometry';
import {createRegionScene, drawLabels} from './support/regions';

const MAXIMUM_REGION_COUNT = 12;

export default defineOperationRun({
  parameters,
  caption: 'Contiguous regions from cutting the spanning tree.',
  createInputs: ({world}) => createRegionScene(makeRandom(7), world, 8, 2),
  build(builder, scene) {
    const {tracts, attributes, columnCount} = scene;
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const values = builder.upload('attributes', 'float32', attributes);
    const flags = builder.allocate('flags', 'uint32', tracts.contiguity.neighbors.length);
    const components = builder.allocate('components', 'uint32', tracts.count);
    const standardized = builder.allocate('standardized', 'float32', attributes.length);
    builder.graph.add(
      new GPUSpatialWeightsMinimumSpanningTree({
        weights,
        values: values.view,
        columnCount,
        standardize: true,
        output: {
          treeEdgeFlags: flags.view,
          componentLabels: components.view,
          standardizedValues: standardized.view
        }
      })
    );
    const parameterBuffer = builder.allocate('parameters', 'uint32', 2);
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    builder.graph.add(
      new GPUSkaterRegions({
        weights,
        treeEdgeFlags: flags.view,
        componentLabels: components.view,
        values: standardized.view,
        columnCount,
        maximumRegionCount: MAXIMUM_REGION_COUNT,
        parameters: parameterBuffer.view,
        output: {
          labels: labels.view,
          regionCount: builder.allocate('region-count', 'uint32', 1).view,
          cutEdges: builder.allocate('cut-edges', 'uint32', MAXIMUM_REGION_COUNT - 1).view,
          cutGains: builder.allocate('cut-gains', 'float32', MAXIMUM_REGION_COUNT - 1).view
        }
      })
    );
    return {
      outputs: {labels, ...uploadTractMesh(builder, tracts)},
      update: current =>
        parameterBuffer.write(
          getGPUSkaterRegionsParameterValues({
            regionCount: current.regionCount,
            minimumRegionSize: current.minimumRegionSize
          })
        )
    };
  },
  draw: ({inputs, outputs}) => drawLabels(inputs, outputs, outputs.labels)
});
