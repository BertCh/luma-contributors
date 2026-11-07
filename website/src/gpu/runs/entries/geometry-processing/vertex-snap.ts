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
import {addJitter, createPolygonSet, uploadPolygonSet} from './shared/polygon-set';
import {addKernel} from './shared/shape-segments';
import {parameters} from './vertex-snap.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Noisy vertices pulled onto the reference vertices within the tolerance.',
  createInputs: ({seed, world}) => ({seed, set: createPolygonSet(makeRandom(seed), world)}),
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
      outputs: {...polygons, noisy, snapped, didSnap},
      update: values => {
        noisy.write(
          addJitter(inputs.set.positions, makeRandom(inputs.seed + 1), values.noise * 0.03)
        );
        parameterBuffer.write(getGPUVertexSnapParameterValues({tolerance: values.tolerance}));
      }
    };
  },
  draw: ({outputs, theme}) => [
    {layer: 'segments', context: true, nodes: outputs.noisy, edges: outputs.edges},
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
