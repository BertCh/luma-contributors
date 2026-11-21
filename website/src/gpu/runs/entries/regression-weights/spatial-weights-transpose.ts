// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsTranspose of directed k-nearest-neighbor weights. Every point has k outgoing
// links; the transpose lists the incoming ones, so points colored by their transposed row length
// are the hubs that many others pick as neighbor.

import {GPUSpatialWeightsTranspose} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints, createKnnWeights} from '../../inputs';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './spatial-weights-transpose.parameters';
import {
  addCardinality,
  addLinkSegments,
  allocateWeights,
  getWeightsInput,
  createSeededRandom,
  getSegmentTargets,
  uploadWeightsBuffers,
  stretchPoints
} from './shared/weights';
import {getArrowGlyphs} from './shared/probes';

export default defineOperationRun({
  parameters,
  caption: 'Points colored by in-degree: the length of their row in the transpose.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    focus: {row: -1},
    index: null as ReturnType<typeof createPointIndex> | null,
    positions: new Float32Array(0) as Float32Array
  }),
  build(builder, inputs, values) {
    const {seed, world} = inputs;
    const points = stretchPoints(
      createClusteredPoints(createSeededRandom(seed), world, 180, 5, values['scatter']),
      world
    );
    inputs.positions = points.positions;
    inputs.index = createPointIndex(points.positions);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const directed = uploadWeightsBuffers(
      builder,
      'directed',
      createKnnWeights(points.positions, values['k'])
    );
    const transposed = allocateWeights(
      builder,
      'transposed',
      points.count,
      directed.capacity,
      true
    );
    builder.graph.add(
      new GPUSpatialWeightsTranspose({
        weights: directed.input,
        output: getWeightsInput(transposed)
      })
    );
    return {
      outputs: {
        positions,
        directedLinks: addLinkSegments(builder, 'directed', directed, positions),
        inDegree: addCardinality(builder, 'transposed', transposed),
        links: addLinkSegments(builder, 'transposed', transposed, positions)
      }
    };
  },
  // The pointer picks a point: arrows leave it along its k links and arrive along the ones the transpose lists.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      if (!inputs.index) return {};
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const outgoing = getSegmentTargets(peek(outputs.directedLinks), from);
      const incoming = getSegmentTargets(peek(outputs.links), from);
      return {
        focus: row,
        glyphs: [
          ...getArrowGlyphs(from, outgoing, 'outgoing'),
          ...getArrowGlyphs(from, incoming, 'incoming'),
          {shape: 'ring', at: from, radius: 0.012}
        ],
        readout: `point ${row} · out ${outgoing.length} · in ${incoming.length}${incoming.length > 2 * outgoing.length ? ' (hub)' : ''}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'lines',
      segments: outputs.links,
      color: [0.6, 0.65, 0.85, 0.25],
      radius: 1.1
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.inDegree,
      radius: 3.6
    }
  ]
});
