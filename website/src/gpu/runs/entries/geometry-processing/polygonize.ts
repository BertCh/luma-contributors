// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPolygonize turns crossing linework into the bounded faces it encloses. The muted lines are
// the input; each face ring the operation wrote is drawn in its own color.

import {GPUPolygonize} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createCrossingLines} from './shared/polygon-set';
import {addOutputEdges, addRingCategories} from './shared/shape-segments';
import {createTopologyBuffers} from './shared/topology-output';
import {parameters} from './polygonize.parameters';

export default defineOperationRun({
  parameters,
  caption: 'The bounded faces enclosed by the lines, one color per face.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, inputs, values) {
    const lines = createCrossingLines(
      makeRandom(inputs.seed),
      inputs.world,
      values.firstCount,
      values.secondCount,
      10,
      0.05
    );
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const topology = createTopologyBuffers(builder, {segments: 512, rings: 64, vertices: 1024});
    builder.graph.add(
      new GPUPolygonize({
        lines: {kind: 'lines', positions: positions.view, lineOffsets: lineOffsets.view},
        intersectionCapacity: 512,
        precision: {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'},
        output: topology.output
      })
    );
    return {
      outputs: {
        faces: topology.positions,
        edges: addOutputEdges(
          builder,
          'faces',
          topology.ringOffsets,
          topology.vertexCapacity,
          true
        ),
        categories: addRingCategories(
          builder,
          'faces',
          topology.ringOffsets,
          topology.vertexCapacity
        ),
        inputSegments: builder.upload(
          'input-segments',
          'float32x4',
          Float32Array.from([...lines.firstSegments, ...lines.secondSegments])
        )
      }
    };
  },
  rebuildOn: ['firstCount', 'secondCount'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.inputSegments},
    {
      layer: 'segments',
      nodes: outputs.faces,
      edges: outputs.edges,
      values: outputs.categories,
      radius: 2.6
    }
  ]
});
