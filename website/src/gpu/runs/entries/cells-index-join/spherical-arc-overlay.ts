// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSphericalArcOverlay combines two spherical regions given by simple boundary rings, using exact
// great-circle arc arrangement. The thin outlines are the operands; the bright ring is the open
// ring assembly the operation wrote (unit vectors converted to longitude/latitude to draw).

import {GPUSphericalArcOverlay} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {addCountedRingSegments, addUnitVectorsToLonLat} from './support/kernels';
import {packUnitVectors} from './support/polygons';
import {
  getLakeShore,
  getRingSegments,
  getSurveySwath,
  SUPERIOR_WINDOW
} from './support/spherical-surveys';
import {parameters} from './spherical-arc-overlay.parameters';

const OPERATIONS = ['intersection', 'union', 'difference', 'symmetric-difference'] as const;
const OUTPUT_VERTICES = 128;
const LONGITUDE_PARAMETER = findParameter(parameters, 'shift');
const LATITUDE_PARAMETER = findParameter(parameters, 'shiftLatitude');

export default defineOperationRun({
  parameters,
  geographicWindow: SUPERIOR_WINDOW,
  caption:
    'Lake Superior’s mapped shoreline and a modeled survey corridor, combined along great-circle arcs.',
  createInputs: () => ({shore: getLakeShore('Lake Superior')}),
  build(builder, {shore: leftRing}, values) {
    const rightVertexCount = 4;
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
      builder.upload('left-outline', 'float32x4', getRingSegments([leftRing])),
      builder.allocate('right-outline', 'float32x4', rightVertexCount)
    ];
    return {
      outputs: {
        operandOutlines,
        counts,
        result: addCountedRingSegments(builder, 'result', {positions: lonLat, ringOffsets, counts})
      },
      update: current => {
        const ring = getSurveySwath(current.shift, current.shiftLatitude);
        rightPositions.write(packUnitVectors([ring]));
        operandOutlines[1].write(getRingSegments([ring]));
      }
    };
  },
  rebuildOn: ['operation'],
  // Drag: the survey corridor is centered on the pointer's longitude and latitude.
  interaction: {
    kind: 'drag',
    hint: 'drag the corridor',
    rest: ({toUnit}) => toUnit([LONGITUDE_PARAMETER.default, LATITUDE_PARAMETER.default]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const shift = snapToParameter(LONGITUDE_PARAMETER, pointer.x);
      const shiftLatitude = snapToParameter(LATITUDE_PARAMETER, pointer.y);
      const counts = peek(outputs.counts);
      return {
        values: {shift, shiftLatitude},
        glyphs: [{shape: 'crosshair', at: [shift, shiftLatitude]}],
        readout: `${OPERATIONS[values.operation]} · ${counts ? `${counts[1]} rings, ${counts[0]} vertices` : '…'}`
      };
    }
  },
  draw: ({outputs, theme}) => [
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
    {layer: 'lines', segments: outputs.result, radius: 3, color: [...theme.ink, 1]}
  ]
});
