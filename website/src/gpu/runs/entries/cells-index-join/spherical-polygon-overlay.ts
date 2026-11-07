// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSphericalPolygonOverlay computes boolean operations between spherical polygons with holes. The
// thin outlines are the two operands; the bright rings are the result polygon, read from the packed
// GeoArrow-style output and converted from unit vectors to longitude/latitude to draw.

import {GPUSphericalPolygonOverlay} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addRingOffsetSegments, addUnitVectorsToLonLat} from './support/kernels';
import {getUnitVector, packUnitVectors} from './support/polygons';
import {parameters} from './spherical-polygon-overlay.parameters';

const OPERATIONS = ['intersection', 'union', 'difference', 'symmetric-difference'] as const;
const OUTPUT_VERTICES = 256;
const OUTPUT_OFFSETS = 33;

function createStar(center: [number, number], radius: number, points: number) {
  return Array.from({length: points * 2}, (_, index) => {
    const angle = (Math.PI * index) / points + 0.2;
    const length = index % 2 === 0 ? radius : radius * 0.55;
    return [center[0] + length * Math.cos(angle), center[1] + length * Math.sin(angle)];
  });
}

function createCircle(center: [number, number], radius: number, count: number) {
  return Array.from({length: count}, (_, index) => {
    const angle = (2 * Math.PI * index) / count + 0.6;
    return [center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)];
  });
}

export default defineOperationRun({
  parameters,
  caption: 'Boolean overlay of a star with a hole and a circle on the sphere.',
  createInputs: ({world}) => ({center: [world.width / 2, 0.5] as [number, number]}),
  build(builder, inputs, values) {
    const leftRings = [createStar(inputs.center, 0.4, 5), createCircle(inputs.center, 0.1, 8)];
    const leftOffsets = [0, leftRings[0].length, leftRings[0].length + leftRings[1].length];
    const left = {
      kind: 'polygons' as const,
      positions: builder.upload('left-positions', 'float32x3', packUnitVectors(leftRings)),
      ringOffsets: builder.upload('left-rings', 'uint32', Uint32Array.from(leftOffsets)),
      polygonOffsets: builder.upload('left-polygons', 'uint32', Uint32Array.of(0, 2)),
      featureOffsets: builder.upload('left-features', 'uint32', Uint32Array.of(0, 1))
    };
    const rightVertexCount = 24;
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
        operation: OPERATIONS[values.operation],
        chartCenter: getUnitVector(inputs.center[0], inputs.center[1]),
        vertexTolerance: 2 ** -18,
        capacity: {intersections: 256, nodedSegments: 512},
        output: {
          geometry: {
            kind: 'polygons',
            positions: positions.view,
            ringOffsets: ringOffsets.view,
            polygonOffsets: polygonOffsets.view,
            featureOffsets: featureOffsets.view,
            sourceIds: builder.allocate('source-ids', 'uint32', OUTPUT_OFFSETS - 1).view
          },
          complement: status[3].view,
          status: {
            count: count.view,
            requiredCount: status[0].view,
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
    const result = addRingOffsetSegments(builder, 'result', {
      positions: lonLat,
      featureOffsets,
      polygonOffsets,
      ringOffsets,
      count
    });
    const operandOutlines = [
      builder.upload('left-outline', 'float32x4', ringsToSegments(leftRings)),
      builder.allocate('right-outline', 'float32x4', rightVertexCount)
    ];
    return {
      outputs: {result, operandOutlines},
      update: current => {
        const ring = createCircle(
          [inputs.center[0] + current.shift, inputs.center[1] + 0.05],
          0.3,
          rightVertexCount
        );
        rightPositions.write(packUnitVectors([ring]));
        operandOutlines[1].write(ringsToSegments([ring]));
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

function ringsToSegments(rings: number[][][]): Float32Array {
  return Float32Array.from(
    rings.flatMap(ring =>
      ring.flatMap(([x, y], index) => [x, y, ...ring[(index + 1) % ring.length]])
    )
  );
}
