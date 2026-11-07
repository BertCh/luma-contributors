// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMakeValid rebuilds valid polygons from rings that cross themselves. Some input stars are
// bow-tied (muted outlines cross); the rings the operation wrote are drawn one color per ring.

import {GPUMakeValid} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, injectDefects, uploadPolygonSet} from './shared/polygon-set';
import {addOutputEdges, addRingCategories} from './shared/shape-segments';
import {createTopologyBuffers} from './shared/topology-output';
import {parameters} from './make-valid.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Bow-tied polygons split into valid rings.',
  createInputs: ({seed, world}) => ({
    set: createPolygonSet(makeRandom(seed), world, {
      rows: 2,
      columns: 4,
      vertexCount: 8,
      irregularity: 0.25,
      closeRings: true
    })
  }),
  build(builder, {set}) {
    const polygons = uploadPolygonSet(builder, set);
    const topology = createTopologyBuffers(builder, {segments: 512, rings: 64, vertices: 1024});
    builder.graph.add(
      new GPUMakeValid({
        polygons: {
          kind: 'polygons',
          positions: polygons.positions.view,
          featureOffsets: polygons.featureOffsets.view,
          polygonOffsets: polygons.polygonOffsets.view,
          ringOffsets: polygons.ringOffsets.view
        },
        intersectionCapacity: 512,
        precision: {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'},
        output: topology.output
      })
    );
    return {
      outputs: {
        ...polygons,
        valid: topology.positions,
        validEdges: addOutputEdges(
          builder,
          'valid',
          topology.ringOffsets,
          topology.vertexCapacity,
          true
        ),
        categories: addRingCategories(
          builder,
          'valid',
          topology.ringOffsets,
          topology.vertexCapacity
        )
      },
      update: values => polygons.positions.write(injectDefects(set, values.defects, ['bow-tie']))
    };
  },
  draw: ({outputs}) => [
    {layer: 'segments', context: true, nodes: outputs.positions, edges: outputs.edges},
    {
      layer: 'segments',
      nodes: outputs.valid,
      edges: outputs.validEdges,
      values: outputs.categories,
      radius: 2.2
    }
  ]
});
