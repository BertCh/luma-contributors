// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUShapeGenerator writes a circle, sector or ellipse ring around every center. The segment count
// and radius scale are per-frame parameters, so smoothness changes without recompiling. Rings are
// colored by feature; dots are the vertices the operation wrote, squares of small dots the centers.

import {
  getGPUShapeGeneratorParameterValues,
  GPUShapeGenerator,
  getGPUShapeVertexCount
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {sampleVesselTracks} from '../../../data/vessel-tracks';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {
  findNearestVertex,
  findRestAround,
  addPackedSegments,
  HIGHLIGHT_COLOR
} from './shared/lines';
import {parameters} from './shape-generator.parameters';

const SHAPES = ['circle', 'sector', 'ellipse'] as const;
const FEATURE_COUNT = 8;
const MAXIMUM_SEGMENTS = 48;
const BASE_RADIUS = 0.1;
// Pointer distance from a vessel that maps to the largest radiusScale.
const REACH = 0.4;

export default defineOperationRun({
  parameters,
  caption:
    'Modeled search areas around the last San Francisco Bay AIS fixes. Sectors follow the last observed course.',
  createInputs: ({world}) => {
    const fleet = sampleVesselTracks(world, FEATURE_COUNT, 2, 72);
    const centers = new Float32Array(FEATURE_COUNT * 2);
    const radii = new Float32Array(FEATURE_COUNT).fill(BASE_RADIUS);
    const ellipseRadii = new Float32Array(FEATURE_COUNT * 2);
    const bearings = new Float32Array(FEATURE_COUNT * 2);
    const rotations = new Float32Array(FEATURE_COUNT);
    for (let vessel = 0; vessel < FEATURE_COUNT; vessel++) {
      const first = fleet.trackOffsets[vessel];
      const last = fleet.trackOffsets[vessel + 1] - 1;
      const x = fleet.points[last * 2];
      const y = fleet.points[last * 2 + 1];
      let previous = last - 1;
      // A vessel can transmit several stationary fixes before the end of its track.
      while (
        previous > first &&
        fleet.points[previous * 2] === x &&
        fleet.points[previous * 2 + 1] === y
      ) {
        previous--;
      }
      const course =
        (Math.atan2(x - fleet.points[previous * 2], y - fleet.points[previous * 2 + 1]) * 180) /
        Math.PI;
      centers.set([x, y], vessel * 2);
      ellipseRadii.set([0.14, 0.055], vessel * 2);
      bearings.set([course - 45, course + 45], vessel * 2);
      rotations[vessel] = course - 90;
    }
    return {centers, radii, ellipseRadii, bearings, rotations, tracks: fleet.segments, world};
  },
  build(builder, inputs, values) {
    const shape = SHAPES[values.shape];
    const maximumVertices = getGPUShapeVertexCount(shape, MAXIMUM_SEGMENTS);
    const centers = builder.upload('centers', 'float32x2', inputs.centers);
    const radii = builder.upload('radii', 'float32', inputs.radii);
    const ellipseRadii = builder.upload('ellipse-radii', 'float32x2', inputs.ellipseRadii);
    const bearings = builder.upload('bearings', 'float32x2', inputs.bearings);
    const rotations = builder.upload('rotations', 'float32', inputs.rotations);
    const positions = builder.allocate(
      'ring-positions',
      'float32x2',
      FEATURE_COUNT * maximumVertices
    );
    const offsets = builder.allocate('ring-offsets', 'uint32', FEATURE_COUNT + 1);
    const parameterBuffer = builder.allocate('shape-parameters', 'float32', 2);
    const common = {
      centers: centers.view,
      maximumSegments: MAXIMUM_SEGMENTS,
      parameters: parameterBuffer.view,
      output: {positions: positions.view, offsets: offsets.view}
    };
    builder.graph.add(
      new GPUShapeGenerator(
        shape === 'ellipse'
          ? {shape, radii: ellipseRadii.view, rotations: rotations.view, ...common}
          : shape === 'sector'
            ? {shape, radii: radii.view, bearings: bearings.view, ...common}
            : {shape, radii: radii.view, ...common}
      )
    );
    const drawing = addPackedSegments(builder, 'rings', positions, offsets, {
      closed: shape !== 'sector'
    });
    return {
      outputs: {
        centers,
        tracks: builder.upload('observed-tracks', 'float32x4', inputs.tracks),
        ...drawing
      },
      update: current =>
        parameterBuffer.write(
          getGPUShapeGeneratorParameterValues({
            segmentCount: current.segmentCount,
            radiusScale: current.radiusScale
          })
        )
    };
  },
  rebuildOn: ['shape'],
  // The distance from the pointer to the nearest vessel sets radiusScale, spread over the whole
  // picture (about 0.45 world units reaches the largest ring), so every position is live.
  interaction: {
    kind: 'reach',
    hint: 'move off a vessel',
    rest: ({inputs, toUnit}) => {
      const scale = findParameter(parameters, 'radiusScale');
      return toUnit(
        findRestAround(
          [inputs.centers[0], inputs.centers[1]],
          ((scale.default - scale.min) / (scale.max - scale.min)) * REACH,
          (x, y) => findNearestVertex(inputs.centers, x, y)?.distance ?? 0,
          inputs.world
        )
      );
    },
    resolve(pointer, {inputs, values}) {
      const hit = findNearestVertex(inputs.centers, pointer.x, pointer.y);
      if (!hit) return {};
      const scale = findParameter(parameters, 'radiusScale');
      const radiusScale = snapToParameter(
        scale,
        scale.min + (scale.max - scale.min) * Math.min(1, hit.distance / REACH)
      );
      return {
        values: {radiusScale},
        glyphs: [
          {shape: 'line', from: hit.point, to: [pointer.x, pointer.y], dashed: true},
          {shape: 'dot', at: hit.point}
        ],
        readout: `radiusScale ${radiusScale.toFixed(2)} · ${values.segmentCount} segments`
      };
    }
  },
  draw: ({outputs, values}) => [
    {layer: 'lines', segments: outputs.tracks, context: true, radius: 0.9, alpha: 0.6},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2.4
    },
    ...(values.segmentCount <= 16
      ? [{layer: 'dots' as const, positions: outputs.points, color: HIGHLIGHT_COLOR, radius: 2.6}]
      : []),
    {layer: 'dots', context: true, positions: outputs.centers, radius: 3}
  ]
});
