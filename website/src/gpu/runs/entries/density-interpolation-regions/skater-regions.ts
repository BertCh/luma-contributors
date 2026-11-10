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
import {bindAxes} from '../../interaction';
import {
  addFlaggedSegments,
  addSegmentGather,
  createRegionScene,
  drawLabels
} from './support/regions';

const MAXIMUM_REGION_COUNT = 12;

export default defineOperationRun({
  parameters,
  caption:
    'Contiguous regions from cutting the spanning tree (white), cut where the red edges fall.',
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
    const cutEdges = builder.allocate('cut-edges', 'uint32', MAXIMUM_REGION_COUNT - 1);
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
          cutEdges: cutEdges.view,
          cutGains: builder.allocate('cut-gains', 'float32', MAXIMUM_REGION_COUNT - 1).view
        }
      })
    );
    // One segment per adjacency slot between the tract centers; the tree flags pick the tree edges.
    const slotSegments = new Float32Array(tracts.contiguity.neighbors.length * 4);
    for (let row = 0; row < tracts.count; row++) {
      for (
        let slot = tracts.contiguity.offsets[row];
        slot < tracts.contiguity.offsets[row + 1];
        slot++
      ) {
        const neighbor = tracts.contiguity.neighbors[slot];
        slotSegments.set(
          [
            tracts.centers[row * 2],
            tracts.centers[row * 2 + 1],
            tracts.centers[neighbor * 2],
            tracts.centers[neighbor * 2 + 1]
          ],
          slot * 4
        );
      }
    }
    const slotBuffer = builder.upload('slot-segments', 'float32x4', slotSegments);
    const tree = addFlaggedSegments(builder, flags, slotBuffer, 'tree');
    // The edges the greedy steps removed, in order: the tree falls apart along these.
    const cuts = addSegmentGather(builder, cutEdges, slotBuffer, 'cuts');
    return {
      outputs: {labels, tree, cuts, ...uploadTractMesh(builder, tracts)},
      update: current =>
        parameterBuffer.write(
          getGPUSkaterRegionsParameterValues({
            regionCount: current.regionCount,
            minimumRegionSize: current.minimumRegionSize
          })
        )
    };
  },
  // x cuts the tree into more regions, y raises the smallest region allowed.
  interaction: bindAxes(parameters, {
    kind: 'pad',
    hint: 'cut the tree',
    x: 'regionCount',
    y: 'minimumRegionSize'
  }),
  draw: ({inputs, outputs}) => [
    ...drawLabels(inputs, outputs, outputs.labels),
    {
      layer: 'lines',
      segments: outputs.tree,
      count: inputs.tracts.contiguity.neighbors.length,
      radius: 1.4,
      color: [1, 1, 1, 0.9]
    },
    {layer: 'lines', segments: outputs.cuts, radius: 3.2, color: [0.95, 0.2, 0.2, 1]}
  ]
});
