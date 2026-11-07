// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPolygonPrecisionReduction rounds polygons to a grid and rebuilds them valid. The muted
// outlines are the input stars; the rings the operation wrote are drawn one color per ring.

import {GPUPolygonPrecisionReduction} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet} from './shared/polygon-set';
import {addOutputEdges, addRingCategories} from './shared/shape-segments';
import {createTopologyBuffers} from './shared/topology-output';
import {parameters} from './polygon-precision-reduction.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Polygons rounded to the precision grid, rebuilt valid.',
  createInputs: ({seed, world}) => ({
    set: createPolygonSet(makeRandom(seed), world, {
      rows: 2,
      columns: 4,
      vertexCount: 14,
      irregularity: 0.4,
      closeRings: true
    })
  }),
  build(builder, {set}, values) {
    const polygons = uploadPolygonSet(builder, set);
    const topology = createTopologyBuffers(builder, {segments: 1024, rings: 64, vertices: 2048});
    builder.graph.add(
      new GPUPolygonPrecisionReduction({
        polygons: {
          kind: 'polygons',
          positions: polygons.positions.view,
          featureOffsets: polygons.featureOffsets.view,
          polygonOffsets: polygons.polygonOffsets.view,
          ringOffsets: polygons.ringOffsets.view
        },
        gridSize: values.gridSize,
        mode: 'valid-output',
        lattice: builder.allocate('lattice', 'float32', 4).view,
        intersectionCapacity: 512,
        output: topology.output
      })
    );
    return {
      outputs: {
        ...polygons,
        reduced: topology.positions,
        reducedEdges: addOutputEdges(
          builder,
          'reduced',
          topology.ringOffsets,
          topology.vertexCapacity,
          true
        ),
        categories: addRingCategories(
          builder,
          'reduced',
          topology.ringOffsets,
          topology.vertexCapacity
        )
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'segments', context: true, nodes: outputs.positions, edges: outputs.edges},
    {
      layer: 'segments',
      nodes: outputs.reduced,
      edges: outputs.reducedEdges,
      values: outputs.categories,
      radius: 2.4
    }
  ]
});
