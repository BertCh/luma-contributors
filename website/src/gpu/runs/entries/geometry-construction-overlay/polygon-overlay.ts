// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPolygonOverlay combines two polygon layers (left: muted, right: orange outline) with a set
// operation. The ring edges the operation wrote are drawn in color, one color per output ring.

import {
  getPolygonTriangulationIndexCount,
  GPUPolygonOverlay,
  GPUPolygonTriangulation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {HIGHLIGHT_COLOR, OUTPUT_COLOR} from './shared/lines';
import {
  addPolygonDrawing,
  allocateOverlayOutput,
  createBlobSet,
  createRingSegments,
  getPolygonFields,
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
    const getRight = (shiftX: number, shiftY: number) =>
      createBlobSet(makeFixedRandom(), world, {...options, shiftX, shiftY});
    // The pointer's displacement from the center, in cells, added to the slider's shift.
    return {left, getRight, world, drag: [0, 0] as [number, number]};
  },
  build(builder, inputs, values) {
    const left = uploadPolygonSet(builder, 'left', inputs.left);
    const initialRight = inputs.getRight(values.shift, values.shift * 0.6);
    const right = uploadPolygonSet(builder, 'right', initialRight);
    const result = allocateOverlayOutput(builder, 'overlay', {
      vertices: 1024,
      rings: 64,
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
    // Fill: the rings the overlay wrote are triangulated on the GPU, so the result reads as an area.
    const ringVertices = result.polygons.positions.length;
    const polygonCapacity = result.polygons.polygonOffsets.length - 1;
    const triangleCapacity =
      getPolygonTriangulationIndexCount(
        ringVertices,
        result.polygons.ringOffsets.length - 1,
        polygonCapacity
      ) / 3;
    const fillIndices = builder.allocate('fill-indices', 'uint32', triangleCapacity * 3);
    const fillValid = builder.allocate('fill-valid', 'uint32', polygonCapacity);
    const fillOffsets = builder.allocate('fill-offsets', 'uint32', polygonCapacity + 1);
    const fillCounts = builder.allocate('fill-counts', 'uint32', polygonCapacity);
    builder.graph.add(
      new GPUPolygonTriangulation({
        ...getPolygonFields(result.polygons),
        largePolygonMethod: 'monotone',
        output: {
          indices: fillIndices.view,
          valid: fillValid.view,
          triangleCount: fillCounts.view,
          triangleOffsets: fillOffsets.view
        }
      })
    );
    const fillOwners = builder.upload('fill-owners', 'uint32', new Uint32Array(triangleCapacity));
    const fillValues = builder.upload('fill-values', 'float32', new Float32Array(1));
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
      outputs: {
        leftOutline,
        rightOutline,
        fillPositions: result.polygons.positions,
        fillIndices,
        fillOwners,
        fillValues,
        triangleCapacity,
        ...drawing
      },
      update: current => {
        const nextRight = inputs.getRight(
          current.shift + inputs.drag[0],
          current.shift * 0.6 + inputs.drag[1]
        );
        right.positions.write(nextRight.positions);
        rightOutline.write(createRingSegments(nextRight));
      }
    };
  },
  rebuildOn: ['operation'],
  // The pointer carries the right layer: its displacement from the center, in cells, moves every
  // right polygon, and the operation re-runs on the new overlap.
  interaction: {
    kind: 'drag',
    hint: 'drag the orange layer',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, values}) {
      const center: [number, number] = [inputs.world.width / 2, inputs.world.height / 2];
      const cellWidth = inputs.world.width / 3;
      const cellHeight = inputs.world.height / 2;
      inputs.drag = [
        clamp((pointer.x - center[0]) / cellWidth, -0.7, 0.7),
        clamp((pointer.y - center[1]) / cellHeight, -0.7, 0.7)
      ];
      const shiftX = values.shift + inputs.drag[0];
      const shiftY = values.shift * 0.6 + inputs.drag[1];
      return {
        rerun: true,
        glyphs: [
          {shape: 'arrow', from: center, to: [pointer.x, pointer.y]},
          {shape: 'crosshair', at: [pointer.x, pointer.y]}
        ],
        readout: `${OPERATIONS[values.operation]} · shift ${shiftX.toFixed(2)}, ${shiftY.toFixed(2)} cells`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fillPositions,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.fillValues,
      triangleCount: outputs.triangleCapacity,
      color: OUTPUT_COLOR,
      alpha: 0.4
    },
    {layer: 'lines', context: true, segments: outputs.leftOutline, radius: 1.6},
    {
      layer: 'lines',
      segments: outputs.rightOutline,
      color: HIGHLIGHT_COLOR,
      alpha: 0.6,
      radius: 1.2
    },
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 3.4
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
