// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSphericalArcOverlay combines two spherical regions given by simple boundary rings, using exact
// great-circle arc arrangement. The thin outlines are the operands; the bright ring is the open
// ring assembly the operation wrote (unit vectors converted to longitude/latitude to draw).

import {GPUSphericalArcOverlay} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCountedRingSegments, addUnitVectorsToLonLat} from './support/kernels';
import {packUnitVectors} from './support/polygons';
import {parameters} from './spherical-arc-overlay.parameters';

const OPERATIONS = ['intersection', 'union', 'difference', 'symmetric-difference'] as const;
const OUTPUT_VERTICES = 64;

function createRing(center: [number, number], radius: number, count: number, turn: number) {
  return Array.from({length: count}, (_, index) => {
    const angle = (2 * Math.PI * index) / count + turn;
    const length = radius * (index % 2 === 0 ? 1 : 0.8);
    return [center[0] + length * Math.cos(angle), center[1] + length * Math.sin(angle)];
  });
}

function ringToSegments(ring: number[][]): Float32Array {
  return Float32Array.from(
    ring.flatMap(([x, y], index) => [x, y, ...ring[(index + 1) % ring.length]])
  );
}

export default defineOperationRun({
  parameters,
  caption: 'Union, intersection or difference of two spherical rings by exact arc arrangement.',
  createInputs: ({world}) => ({center: [world.width / 2, 0.5] as [number, number]}),
  build(builder, inputs, values) {
    const leftRing = createRing([inputs.center[0] - 0.1, inputs.center[1]], 0.32, 8, 0.1);
    const rightVertexCount = 8;
    const rightPositions = builder.allocate('right-positions', 'float32x3', rightVertexCount);
    const positions = builder.allocate('result-positions', 'float32x3', OUTPUT_VERTICES);
    const ringOffsets = builder.allocate('result-rings', 'uint32', 17);
    const counts = builder.allocate('counts', 'uint32', 4);
    const status = builder.allocate('status', 'uint32', 1);
    builder.graph.add(
      new GPUSphericalArcOverlay({
        left: {
          positions: builder.upload('left-positions', 'float32x3', packUnitVectors([leftRing]))
            .view,
          ringOffsets: builder.upload('left-rings', 'uint32', Uint32Array.of(0, leftRing.length))
            .view,
          referenceInside: false
        },
        right: {
          positions: rightPositions.view,
          ringOffsets: builder.upload('right-rings', 'uint32', Uint32Array.of(0, rightVertexCount))
            .view,
          referenceInside: false
        },
        referencePoint: [0, 0, 1],
        operation: OPERATIONS[values.operation],
        output: {
          positions: positions.view,
          ringOffsets: ringOffsets.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    const lonLat = addUnitVectorsToLonLat(builder, 'result', positions);
    const operandOutlines = [
      builder.upload('left-outline', 'float32x4', ringToSegments(leftRing)),
      builder.allocate('right-outline', 'float32x4', rightVertexCount)
    ];
    return {
      outputs: {
        operandOutlines,
        result: addCountedRingSegments(builder, 'result', {positions: lonLat, ringOffsets, counts})
      },
      update: current => {
        const ring = createRing(
          [inputs.center[0] + current.shift, inputs.center[1] + 0.04],
          0.3,
          rightVertexCount,
          0.5
        );
        rightPositions.write(packUnitVectors([ring]));
        operandOutlines[1].write(ringToSegments(ring));
      }
    };
  },
  rebuildOn: ['operation'],
  draw: ({outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.operandOutlines[0],
      radius: 1.2,
      color: [0.45, 0.65, 1, 0.85]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.operandOutlines[1],
      radius: 1.2,
      color: [1, 0.6, 0.35, 0.85]
    },
    {layer: 'lines', segments: outputs.result, radius: 3, color: [1, 1, 1, 1]}
  ]
});
