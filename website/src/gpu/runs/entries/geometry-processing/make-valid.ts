// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMakeValid rebuilds valid polygons from rings that cross themselves. Some input stars are
// bow-tied (muted outlines cross); the rings the operation wrote are drawn one color per ring.

import {GPUMakeValid} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {
  applyVertexDrag,
  createPolygonSet,
  getDraggedShell,
  injectDefects,
  uploadPolygonSet
} from './shared/polygon-set';
import {
  carryNearestVertex,
  chooseHeroVertex,
  findRingCrossings,
  getHeroRest
} from './shared/pointer-support';
import {addOutputEdges, addRingCategories} from './shared/shape-segments';
import {createTopologyBuffers} from './shared/topology-output';
import type {VertexDrag} from './shared/polygon-set';
import {parameters} from './make-valid.parameters';

const OPTIONS = {rows: 2, columns: 4, vertexCount: 8, irregularity: 0.25, closeRings: true};

export default defineOperationRun({
  parameters,
  caption: 'Bow-tied polygons split into valid rings.',
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, OPTIONS);
    return {seed, world, set, drag: null as VertexDrag | null, hero: chooseHeroVertex(set, 5)};
  },
  build(builder, inputs) {
    const {set} = inputs;
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
        ringOffsets: topology.ringOffsets,
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
      update: values => {
        const next = createPolygonSet(makeRandom(inputs.seed), inputs.world, OPTIONS);
        applyVertexDrag(next, inputs.drag);
        inputs.set = next;
        polygons.positions.write(injectDefects(next, values.defects, ['bow-tie']));
      }
    };
  },
  // The pointer pushes the nearest shell vertex; dragged across the ring it makes a bow-tie.
  interaction: {
    kind: 'drag',
    hint: 'drag a vertex across the ring',
    rest: ({inputs, toUnit}) => toUnit(getHeroRest(inputs)),
    resolve(pointer, {inputs, outputs, peek}) {
      const carried = carryNearestVertex(inputs, pointer, 0, 0.45);
      const crossings = inputs.drag
        ? findRingCrossings(getDraggedShell(inputs.set, inputs.drag), 0, OPTIONS.vertexCount)
        : [];
      const offsets = peek(outputs.ringOffsets);
      let ringCount = 0;
      while (offsets && offsets[ringCount + 1] > offsets[ringCount]) ringCount++;
      return {
        rerun: true,
        glyphs: [...carried.glyphs, ...crossings.map(at => ({shape: 'dot' as const, at}))],
        readout: `${crossings.length} bow-tie crossings · ${offsets ? ringCount : '...'} valid rings`
      };
    }
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
