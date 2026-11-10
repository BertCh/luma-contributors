// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVertexSnap moves each noisy vertex onto the nearest reference vertex within `tolerance`.
// Muted outlines are the noisy input, dots are the reference vertices, and the output edges are
// colored by whether their vertex snapped.

import {
  getGPUVertexSnapParameterValues,
  GPUVertexSnap,
  GPU_VERTEX_SNAP_NO_REFERENCE
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {createPointIndex} from '../../pick';
import {addJitter, createPolygonSet, uploadPolygonSet} from './shared/polygon-set';
import {findPointAtDistance} from './shared/pointer-support';
import {addKernel, addPairSegments} from './shared/shape-segments';
import {parameters} from './vertex-snap.parameters';

const TOLERANCE = findParameter(parameters, 'tolerance');
/** The tolerance is this fraction of the pointer's distance to the nearest reference vertex. */
const DISTANCE_SCALE = 0.4;

export default defineOperationRun({
  parameters,
  caption: 'Noisy vertices pulled onto the reference vertices within the tolerance.',
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {rows: 2, columns: 4});
    const index = createPointIndex(set.positions);
    return {
      seed,
      set,
      index,
      restPoint: findPointAtDistance(
        (x, y) => index.distanceTo(index.nearest(x, y), x, y),
        TOLERANCE.default / DISTANCE_SCALE,
        world
      )
    };
  },
  build(builder, inputs) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const vertexCount = inputs.set.positions.length / 2;
    const noisy = builder.allocate('noisy', 'float32x2', vertexCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const snapped = builder.allocate('snapped', 'float32x2', vertexCount);
    const referenceRows = builder.allocate('reference-rows', 'uint32', vertexCount);
    builder.graph.add(
      new GPUVertexSnap({
        positions: noisy.view,
        referencePositions: polygons.positions.view,
        parameters: parameterBuffer.view,
        output: {positions: snapped.view, referenceRows: referenceRows.view}
      })
    );
    const didSnap = builder.allocate('did-snap', 'uint32', vertexCount);
    addKernel(builder, {
      id: 'snap-flags',
      invocationCount: vertexCount,
      bindings: [
        {name: 'rows', buffer: referenceRows, access: 'read'},
        {name: 'flags', buffer: didSnap, access: 'read_write'}
      ],
      body: `flags[flagsOffset + index] = select(1u, 0u, rows[rowsOffset + index] == ${GPU_VERTEX_SNAP_NO_REFERENCE}u);`
    });
    return {
      outputs: {
        ...polygons,
        noisy,
        snapped,
        didSnap,
        links: addPairSegments(builder, 'snap-links', noisy, snapped)
      },
      update: values => {
        noisy.write(
          addJitter(inputs.set.positions, makeRandom(inputs.seed + 1), values.noise * 0.07)
        );
        parameterBuffer.write(getGPUVertexSnapParameterValues({tolerance: values.tolerance}));
      }
    };
  },
  // The pointer's distance to the nearest reference vertex is the tolerance: its ring passes through the pointer.
  interaction: {
    kind: 'reach',
    hint: 'stretch the snap radius',
    rest: ({inputs, toUnit}) => toUnit(inputs.restPoint),
    resolve(pointer, {inputs, outputs, peek}) {
      const row = inputs.index.nearest(pointer.x, pointer.y);
      const reference: [number, number] = [
        inputs.set.positions[row * 2],
        inputs.set.positions[row * 2 + 1]
      ];
      const tolerance = snapToParameter(
        TOLERANCE,
        inputs.index.distanceTo(row, pointer.x, pointer.y) * DISTANCE_SCALE
      );
      const flags = peek(outputs.didSnap);
      let snapped = 0;
      if (flags) for (const flag of flags) snapped += flag;
      return {
        values: {tolerance},
        glyphs: [
          {shape: 'ring', at: reference, radius: tolerance},
          {shape: 'line', from: reference, to: [pointer.x, pointer.y], dashed: true},
          {shape: 'dot', at: reference}
        ],
        readout: `tolerance ${tolerance.toFixed(3)} · ${flags ? Math.round((100 * snapped) / flags.length) : '...'}% snapped`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {layer: 'segments', context: true, nodes: outputs.noisy, edges: outputs.edges},
    {layer: 'lines', segments: outputs.links, radius: 1, alpha: 0.5},
    {
      layer: 'segments',
      nodes: outputs.snapped,
      edges: outputs.edges,
      values: outputs.didSnap,
      radius: 1.8
    },
    {layer: 'dots', positions: outputs.positions, radius: 2.6, color: [...theme.ink, 1]}
  ]
});
