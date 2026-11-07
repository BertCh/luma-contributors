// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUShapeDescriptors over polygons of varying roundness. Fill flags the slivers (compactness
// below `sliverThreshold`); the dot is the Polsby-Popper compactness (1 is a circle).

import {
  getGPUShapeDescriptorsParameterValues,
  GPUShapeDescriptors
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {parameters} from './shape-descriptors.parameters';

const OPTIONS = {irregularity: 0.12, vertexCount: 16};

export default defineOperationRun({
  parameters,
  caption: 'Slivers flagged by compactness; the dot is the Polsby-Popper score.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    set: createPolygonSet(makeRandom(seed), world, {...OPTIONS, stretch: 0.8})
  }),
  build(builder, inputs) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const polsbyPopper = builder.allocate('polsby-popper', 'float32', inputs.set.count);
    const sliver = builder.allocate('sliver', 'uint32', inputs.set.count);
    builder.graph.add(
      new GPUShapeDescriptors({
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        parameters: parameterBuffer.view,
        output: {polsbyPopper: polsbyPopper.view, sliver: sliver.view}
      })
    );
    return {
      outputs: {...polygons, polsbyPopper, sliver},
      update: values => {
        parameterBuffer.write(
          getGPUShapeDescriptorsParameterValues({sliverThreshold: values.sliverThreshold})
        );
        writePolygonSet(
          polygons,
          createPolygonSet(makeRandom(inputs.seed), inputs.world, {
            ...OPTIONS,
            stretch: values.stretch
          })
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.sliver,
      triangleCount: inputs.set.fillTriangleCount
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {
      layer: 'dots',
      positions: outputs.centers,
      values: outputs.polsbyPopper,
      range: [0, 1],
      radius: 7
    }
  ]
});
