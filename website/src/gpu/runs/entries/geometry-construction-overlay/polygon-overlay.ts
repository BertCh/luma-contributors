// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPolygonOverlay combines two polygon layers (left: muted, right: orange outline) with a set
// operation. The ring edges the operation wrote are drawn in color, one color per output ring.

import {GPUPolygonOverlay} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {HIGHLIGHT_COLOR} from './shared/lines';
import {
  addPolygonDrawing,
  allocateOverlayOutput,
  createBlobSet,
  createRingSegments,
  getPolygonPort,
  uploadPolygonSet
} from './shared/polygons';
import {parameters} from './polygon-overlay.parameters';

const OPERATIONS = ['intersection', 'union', 'difference', 'symmetric-difference'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Two polygon layers combined; the rings the operation wrote are colored.',
  createInputs: ({random, world}) => {
    const options = {rows: 2, columns: 3, vertexCount: 18, irregularity: 0.2, fill: 0.78};
    const left = createBlobSet(random, world, options);
    const getRight = (shift: number) =>
      createBlobSet(makeFixedRandom(), world, {...options, shiftX: shift, shiftY: shift * 0.6});
    return {left, getRight};
  },
  build(builder, inputs, values) {
    const left = uploadPolygonSet(builder, 'left', inputs.left);
    const initialRight = inputs.getRight(values.shift);
    const right = uploadPolygonSet(builder, 'right', initialRight);
    const result = allocateOverlayOutput(builder, 'overlay', {
      vertices: 4096,
      rings: 256,
      boundary: 4096
    });
    builder.graph.add(
      new GPUPolygonOverlay({
        left: getPolygonPort(left),
        right: getPolygonPort(right),
        operation: OPERATIONS[values.operation],
        vertexTolerance: 1e-4,
        capacity: {intersections: 4096, nodedSegments: 16384},
        output: result.output
      })
    );
    const drawing = addPolygonDrawing(builder, 'overlay', result.polygons, result.count);
    const leftOutline = builder.upload(
      'left-outline',
      'float32x4',
      createRingSegments(inputs.left)
    );
    const rightOutline = builder.allocate(
      'right-outline',
      'float32x4',
      createRingSegments(initialRight).length / 4
    );
    return {
      outputs: {leftOutline, rightOutline, ...drawing},
      update: current => {
        const nextRight = inputs.getRight(current.shift);
        right.positions.write(nextRight.positions);
        rightOutline.write(createRingSegments(nextRight));
      }
    };
  },
  rebuildOn: ['operation'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.leftOutline, radius: 3},
    {
      layer: 'lines',
      segments: outputs.rightOutline,
      color: HIGHLIGHT_COLOR,
      alpha: 0.85,
      radius: 1.5
    },
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2.4
    }
  ]
});

// The right layer is regenerated with the same shape every time, so only its position moves.
function makeFixedRandom(): () => number {
  let state = 12345;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}
