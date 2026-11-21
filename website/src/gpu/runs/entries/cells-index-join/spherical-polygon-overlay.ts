// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSphericalPolygonOverlay computes boolean operations between spherical polygons with holes. The
// thin outlines are the two operands; the bright rings are the result polygon, read from the packed
// GeoArrow-style output and converted from unit vectors to longitude/latitude to draw.

import {GPUSphericalPolygonOverlay} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunLayer} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {
  addKernel,
  addRingOffsetSegments,
  addSegmentFill,
  addUnitVectorsToLonLat
} from './support/kernels';
import {
  getSphericalOverlayStatusLabel,
  selectSphericalOverlayLayers,
  SPHERICAL_OVERLAY_DISPLAY_COUNT_WGSL
} from './spherical-polygon-overlay-status';
import {getUnitVector, packUnitVectors} from './support/polygons';
import {
  getLakeShore,
  getRingSegments,
  getSurveySwath,
  SUPERIOR_WINDOW,
  getSurveyRing
} from './support/spherical-surveys';
import {parameters} from './spherical-polygon-overlay.parameters';

const OPERATIONS = ['intersection', 'union', 'difference', 'symmetric-difference'] as const;
const OUTPUT_VERTICES = 256;
const OUTPUT_OFFSETS = 33;
const FILL_COLUMNS = 160;
const FILL_ROWS = 90;
const LONGITUDE_PARAMETER = findParameter(parameters, 'shift');
const LATITUDE_PARAMETER = findParameter(parameters, 'shiftLatitude');

export default defineOperationRun({
  parameters,
  geographicWindow: SUPERIOR_WINDOW,
  caption:
    'Lake Superior with a modeled survey exclusion, clipped against a movable survey corridor.',
  createInputs: () => ({
    rings: [getLakeShore('Lake Superior'), getSurveyRing(-87.7, 47.6, 17, 12).reverse()]
  }),
  build(builder, {rings: leftRings}, values) {
    const leftOffsets = [0, leftRings[0].length, leftRings[0].length + leftRings[1].length];
    const left = {
      kind: 'polygons' as const,
      positions: builder.upload('left-positions', 'float32x3', packUnitVectors(leftRings)),
      ringOffsets: builder.upload('left-rings', 'uint32', Uint32Array.from(leftOffsets)),
      polygonOffsets: builder.upload('left-polygons', 'uint32', Uint32Array.of(0, 2)),
      featureOffsets: builder.upload('left-features', 'uint32', Uint32Array.of(0, 1))
    };
    const rightVertexCount = 4;
    const rightPositions = builder.allocate('right-positions', 'float32x3', rightVertexCount);
    const right = {
      kind: 'polygons' as const,
      positions: rightPositions,
      ringOffsets: builder.upload('right-rings', 'uint32', Uint32Array.of(0, rightVertexCount)),
      polygonOffsets: builder.upload('right-polygons', 'uint32', Uint32Array.of(0, 1)),
      featureOffsets: builder.upload('right-features', 'uint32', Uint32Array.of(0, 1))
    };
    const positions = builder.allocate('result-positions', 'float32x3', OUTPUT_VERTICES);
    const ringOffsets = builder.allocate('result-rings', 'uint32', OUTPUT_OFFSETS);
    const polygonOffsets = builder.allocate('result-polygons', 'uint32', OUTPUT_OFFSETS);
    const featureOffsets = builder.allocate('result-features', 'uint32', OUTPUT_OFFSETS);
    const count = builder.allocate('count', 'uint32', 1);
    const status = ['requiredCount', 'overflow', 'candidateOverflow', 'complement'].map(name =>
      builder.allocate(name, 'uint32', 1)
    );
    const boundaryStatus = ['count', 'requiredCount', 'overflow', 'candidateOverflow'].map(name =>
      builder.allocate(`boundary-${name}`, 'uint32', 1)
    );
    builder.graph.add(
      new GPUSphericalPolygonOverlay({
        left: {
          ...left,
          positions: left.positions.view,
          ringOffsets: left.ringOffsets.view,
          polygonOffsets: left.polygonOffsets.view,
          featureOffsets: left.featureOffsets.view
        },
        right: {
          ...right,
          positions: right.positions.view,
          ringOffsets: right.ringOffsets.view,
          polygonOffsets: right.polygonOffsets.view,
          featureOffsets: right.featureOffsets.view
        },
        operation: OPERATIONS[values['operation']],
        chartCenter: getUnitVector(-88.2, 47.7),
        vertexTolerance: 2 ** -18,
        capacity: {intersections: 256, nodedSegments: 512},
        output: {
          geometry: {
            kind: 'polygons',
            positions: positions.view,
            ringOffsets: ringOffsets.view,
            polygonOffsets: polygonOffsets.view,
            featureOffsets: featureOffsets.view
          },
          complement: status[3].view,
          status: {
            count: count.view,
            requiredRingCount: status[0].view,
            overflow: status[1].view,
            candidateOverflow: status[2].view
          },
          boundary: {
            endpoints: builder.allocate('boundary-endpoints', 'float32x4', 512).view,
            operandIds: builder.allocate('boundary-operands', 'uint32', 512).view,
            sourceFeatureIds: builder.allocate('boundary-sources', 'uint32', 512).view,
            status: {
              count: boundaryStatus[0].view,
              requiredCount: boundaryStatus[1].view,
              overflow: boundaryStatus[2].view,
              candidateOverflow: boundaryStatus[3].view
            }
          }
        }
      })
    );
    const lonLat = addUnitVectorsToLonLat(builder, 'result', positions);
    const displayCount = builder.allocate('complete-result-count', 'uint32', 1);
    addKernel(builder, {
      id: 'complete-result-count',
      invocationCount: 1,
      bindings: [
        {name: 'count', buffer: count, access: 'read'},
        {name: 'overflow', buffer: status[1], access: 'read'},
        {name: 'candidateOverflow', buffer: status[2], access: 'read'},
        {name: 'displayCount', buffer: displayCount, access: 'read_write'}
      ],
      body: SPHERICAL_OVERLAY_DISPLAY_COUNT_WGSL
    });
    const result = addRingOffsetSegments(builder, 'result', {
      positions: lonLat,
      featureOffsets,
      polygonOffsets,
      ringOffsets,
      count: displayCount
    });
    const operandOutlines = [
      builder.upload('left-outline', 'float32x4', getRingSegments(leftRings)),
      builder.allocate('right-outline', 'float32x4', rightVertexCount)
    ];
    return {
      outputs: {
        result,
        fill: addSegmentFill(builder, 'result', result, SUPERIOR_WINDOW, FILL_COLUMNS, FILL_ROWS),
        operandOutlines,
        count,
        overflow: status[1],
        candidateOverflow: status[2],
        featureOffsets,
        polygonOffsets,
        ringOffsets
      },
      update: current => {
        const ring = getSurveySwath(current['shift'], current['shiftLatitude']);
        rightPositions.write(packUnitVectors([ring]));
        operandOutlines[1].write(getRingSegments([ring]));
      }
    };
  },
  rebuildOn: ['operation'],
  async afterRun({outputs}) {
    const [overflow, candidateOverflow] = await Promise.all([
      outputs.overflow.buffer.readAsync(),
      outputs.candidateOverflow.buffer.readAsync()
    ]);
    return {
      overflow: new Uint32Array(overflow.buffer, overflow.byteOffset, 1)[0],
      candidateOverflow: new Uint32Array(
        candidateOverflow.buffer,
        candidateOverflow.byteOffset,
        1
      )[0]
    };
  },
  // Drag: the survey corridor is centered on the pointer's longitude and latitude.
  interaction: {
    kind: 'drag',
    hint: 'drag the corridor',
    rest: ({toUnit}) => toUnit([LONGITUDE_PARAMETER.default, LATITUDE_PARAMETER.default]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const shift = snapToParameter(LONGITUDE_PARAMETER, pointer.x);
      const shiftLatitude = snapToParameter(LATITUDE_PARAMETER, pointer.y);
      const count = peek(outputs.count);
      const featureOffsets = peek(outputs.featureOffsets);
      const polygonOffsets = peek(outputs.polygonOffsets);
      let shape = '…';
      const statusLabel = getSphericalOverlayStatusLabel({
        overflow: peek(outputs.overflow)?.[0],
        candidateOverflow: peek(outputs.candidateOverflow)?.[0]
      });
      if (statusLabel) shape = statusLabel;
      else if (count && featureOffsets && polygonOffsets) {
        const polygons = featureOffsets[count[0]];
        const rings = polygonOffsets[polygons];
        const holes = rings - polygons;
        shape = `${rings} ring${rings === 1 ? '' : 's'} (${holes} hole${holes === 1 ? '' : 's'})`;
      }
      return {
        values: {shift, shiftLatitude},
        glyphs: [{shape: 'crosshair', at: [shift, shiftLatitude]}],
        readout: `${OPERATIONS[values['operation']]} · ${shape}`
      };
    }
  },
  draw: ({outputs, theme, extra}) =>
    selectSphericalOverlayLayers<RunLayer>(
      extra,
      [
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
        }
      ],
      [
        {
          layer: 'raster',
          values: outputs.fill,
          columns: FILL_COLUMNS,
          rows: FILL_ROWS,
          ramp: 'sequential',
          range: [0, 1],
          alpha: 0.4
        },
        {layer: 'lines', segments: outputs.result, radius: 3, color: [...theme.ink, 1]}
      ]
    )
});
