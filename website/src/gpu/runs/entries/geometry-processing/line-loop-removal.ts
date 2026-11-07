// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineLoopRemoval erases the self-loops of a meandering curve. The muted path is the tangled
// input; the bright path is what remains, with the vertex count the operation wrote.

import {GPULineLoopRemoval} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createIdentityIds} from './shared/polygon-set';
import {parameters} from './line-loop-removal.parameters';

const VERTEX_COUNT = 240;

/** A trochoid: `tangle` above 1 curls the curve into loops, below 1 it only waves. */
function createTangledLine(seed: number, world: {width: number; height: number}, tangle: number) {
  const random = makeRandom(seed);
  const positions = new Float32Array(VERTEX_COUNT * 2);
  const turns = 4;
  const radius = (0.84 * world.width) / (turns * 2 * Math.PI);
  for (let vertex = 0; vertex < VERTEX_COUNT; vertex++) {
    const angle = (vertex / (VERTEX_COUNT - 1)) * turns * 2 * Math.PI;
    positions[vertex * 2] =
      0.08 * world.width +
      radius * angle -
      radius * tangle * Math.sin(angle) +
      (random() - 0.5) * 0.004;
    positions[vertex * 2 + 1] =
      world.height * 0.5 - radius * tangle * Math.cos(angle) * 1.2 + (random() - 0.5) * 0.004;
  }
  return positions;
}

export default defineOperationRun({
  parameters,
  caption: 'Loops erased from a meandering curve in drawing order.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, inputs, values) {
    const positions = builder.upload(
      'positions',
      'float32x2',
      createTangledLine(inputs.seed, inputs.world, values.tangle)
    );
    const lineOffsets = builder.upload('line-offsets', 'uint32', Uint32Array.of(0, VERTEX_COUNT));
    const cleaned = builder.allocate('cleaned', 'float32x2', VERTEX_COUNT * 2);
    const cleanedOffsets = builder.allocate('cleaned-offsets', 'uint32', 2);
    const sourceEdges = builder.allocate('source-edges', 'uint32', VERTEX_COUNT * 2);
    const status = builder.allocate('status', 'uint32x4', 1);
    builder.graph.add(
      new GPULineLoopRemoval({
        lines: {kind: 'lines', positions: positions.view, lineOffsets: lineOffsets.view},
        intersectionCapacity: 512,
        output: {
          positions: cleaned.view,
          lineOffsets: cleanedOffsets.view,
          sourceEdges: sourceEdges.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {
        positions,
        cleaned,
        status,
        ids: builder.upload('ids', 'uint32', createIdentityIds(VERTEX_COUNT * 2))
      },
      update: current =>
        positions.write(createTangledLine(inputs.seed, inputs.world, current.tangle))
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'paths',
      context: true,
      nodes: outputs.positions,
      ids: outputs.ids,
      count: VERTEX_COUNT
    },
    {layer: 'paths', nodes: outputs.cleaned, ids: outputs.ids, count: outputs.status, radius: 2.6}
  ]
});
