// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULonLatPolygonPreparation makes longitude/latitude polygons planar-safe: it densifies
// great-circle edges, splits a polygon at the antimeridian and caps a pole-enclosing ring. The
// muted outlines are the raw input on an equirectangular map (the seam-crossing box and the polar
// triangle look wrong there); the colored rings are the prepared output, one color per ring.

import {
  getGPULonLatPolygonPreparationParameterValues,
  GPULonLatPolygonPreparation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import sample from '../../../data/natural-earth-geometry.json';
import {createCoastlineSegments} from '../../../data/coastline';
import {defineOperationRun, type RunBuffer, type RunBuilder} from '../../define-run';
import {clamp} from '../../interaction';
import {createRingEdges} from './shared/polygon-set';
import {addKernel, addOutputEdges, addRingCategories} from './shared/shape-segments';
import {parameters} from './lon-lat-polygon-preparation.parameters';

// Modeled survey bounds in the Bering Sea and around the Arctic pole, plus the mapped Greenland shore.
const BERING_WINDOW = [
  [170, 50],
  [-155, 50],
  [-155, 65],
  [170, 65]
];
const POLAR_WINDOW = [
  [-150, 70],
  [-30, 70],
  [90, 70]
];
const RINGS = [BERING_WINDOW, POLAR_WINDOW, sample.greenlandShore];
const POSITIONS = Float32Array.from(RINGS.flat(2));
const RING_OFFSETS = Uint32Array.of(
  0,
  BERING_WINDOW.length,
  BERING_WINDOW.length + POLAR_WINDOW.length,
  POSITIONS.length / 2
);
const POLYGON_OFFSETS = Uint32Array.of(0, 1, 2, 3);
const FEATURE_OFFSETS = Uint32Array.of(0, 1, 2, 3);
const OUTPUT_VERTICES = 4096;
/** The map is cropped to the northern latitudes where all three rings live. */
const SOUTH = 40;
const NORTH = 90;
const HALF_WIDTH = 17.5;
const BERING_CENTER = [187.5 - 360, 57.5] as const;

const wrapLongitude = (longitude: number) => ((((longitude + 180) % 360) + 360) % 360) - 180;
const getMapX = (longitude: number, width: number) =>
  width * 0.04 + ((longitude + 180) / 360) * width * 0.92;
const getMapY = (latitude: number) => 0.04 + ((latitude - SOUTH) / (NORTH - SOUTH)) * 0.92;

/** Rewrites the Bering window as a box of fixed size around `center` (longitude, latitude). */
function createPositions(center: readonly [number, number]): Float32Array {
  const positions = Float32Array.from(POSITIONS);
  const west = wrapLongitude(center[0] - HALF_WIDTH);
  const east = wrapLongitude(center[0] + HALF_WIDTH);
  const south = center[1] - 7.5;
  const north = center[1] + 7.5;
  positions.set([west, south, east, south, east, north, west, north]);
  return positions;
}

export default defineOperationRun({
  parameters,
  caption:
    'Greenland’s shore densified, a Bering survey window split at the date line and a polar survey ring capped.',
  createInputs: ({world}) => ({world, center: [...BERING_CENTER] as [number, number]}),
  build(builder, inputs, values) {
    const {world} = inputs;
    const positions = builder.upload('positions', 'float32x2', createPositions(inputs.center));
    const featureOffsets = builder.upload('feature-offsets', 'uint32', FEATURE_OFFSETS);
    const polygonOffsets = builder.upload('polygon-offsets', 'uint32', POLYGON_OFFSETS);
    const ringOffsets = builder.upload('ring-offsets', 'uint32', RING_OFFSETS);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const prepared = builder.allocate('prepared', 'float32x2', OUTPUT_VERTICES);
    const preparedRings = builder.allocate('prepared-rings', 'uint32', 2 * RING_OFFSETS.length - 1);
    builder.graph.add(
      new GPULonLatPolygonPreparation({
        positions: positions.view,
        featureOffsets: featureOffsets.view,
        polygonOffsets: polygonOffsets.view,
        ringOffsets: ringOffsets.view,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'none',
          edgeModel: 'great-circle'
        },
        parameters: parameterBuffer.view,
        maximumPieceCount: 1024,
        antimeridian: values.antimeridian === 1 ? 'reject' : 'split',
        poleRings: values.poleRings === 1 ? 'reject' : 'cap',
        output: {
          positions: prepared.view,
          featureOffsets: builder.allocate('prepared-features', 'uint32', FEATURE_OFFSETS.length)
            .view,
          polygonOffsets: builder.allocate(
            'prepared-polygons',
            'uint32',
            2 * POLYGON_OFFSETS.length - 1
          ).view,
          ringOffsets: preparedRings.view,
          ringFlags: builder.allocate('ring-flags', 'uint32', RING_OFFSETS.length - 1).view,
          status: {
            count: builder.allocate('count', 'uint32', 1).view,
            overflow: builder.allocate('overflow', 'uint32', 1).view
          }
        }
      })
    );
    return {
      outputs: {
        coast: builder.upload(
          'coast',
          'float32x4',
          createCoastlineSegments(
            [(world.width * 0.92) / 360, 0.92 / (NORTH - SOUTH)],
            [world.width / 2, 0.04 - (SOUTH * 0.92) / (NORTH - SOUTH)]
          )
        ),
        input: addCroppedMap(builder, 'input', positions, world),
        inputEdges: builder.upload('input-edges', 'uint32x2', createRingEdges(RING_OFFSETS)),
        prepared: addCroppedMap(builder, 'prepared', prepared, world),
        preparedEdges: addOutputEdges(builder, 'prepared', preparedRings, OUTPUT_VERTICES, true),
        categories: addRingCategories(builder, 'prepared', preparedRings, OUTPUT_VERTICES)
      },
      update: current => {
        parameterBuffer.write(
          getGPULonLatPolygonPreparationParameterValues('great-circle', {
            maximumSegmentDegrees: current.maximumSegmentDegrees
          })
        );
        positions.write(createPositions(inputs.center));
      }
    };
  },
  // The pointer carries the Bering survey window; across the date line it is split in two.
  interaction: {
    kind: 'drag',
    hint: 'drag across the date line',
    rest: ({inputs, toUnit}) =>
      toUnit([getMapX(BERING_CENTER[0], inputs.world.width), getMapY(BERING_CENTER[1])]),
    resolve(pointer, {inputs, world, values}) {
      const longitude = clamp(
        ((pointer.x - world.width * 0.04) / (world.width * 0.92)) * 360 - 180,
        -180,
        180
      );
      const latitude = clamp(
        SOUTH + ((pointer.y - 0.04) / 0.92) * (NORTH - SOUTH),
        SOUTH + 8,
        NORTH - 8
      );
      inputs.center = [longitude, latitude];
      const west = wrapLongitude(longitude - HALF_WIDTH);
      const east = wrapLongitude(longitude + HALF_WIDTH);
      const crosses = west > east;
      const edgeX = [getMapX(-180, world.width), getMapX(180, world.width)];
      return {
        rerun: true,
        glyphs: [
          ...edgeX.map(x => ({
            shape: 'line' as const,
            from: [x, 0] as const,
            to: [x, 1] as const,
            dashed: true
          })),
          {shape: 'crosshair', at: [pointer.x, pointer.y]}
        ],
        readout: `window ${Math.round(west)}..${Math.round(east)}° · ${crosses ? (values.antimeridian === 1 ? 'rejected' : 'split in 2') : 'whole'}`
      };
    }
  },
  rebuildOn: ['antimeridian', 'poleRings'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.coast, alpha: 0.4, radius: 0.8},
    {layer: 'segments', context: true, nodes: outputs.input, edges: outputs.inputEdges},
    {
      layer: 'segments',
      nodes: outputs.prepared,
      edges: outputs.preparedEdges,
      values: outputs.categories,
      radius: 2.4
    }
  ]
});

/** Maps longitude/latitude degrees to the drawing world, cropped to the northern latitudes. */
function addCroppedMap(
  builder: RunBuilder,
  id: string,
  lonLat: RunBuffer<'float32x2'>,
  world: {width: number; height: number}
): RunBuffer<'float32x2'> {
  const projected = builder.allocate(`${id}-world`, 'float32x2', lonLat.length);
  addKernel(builder, {
    id: `${id}-cropped-map`,
    invocationCount: lonLat.length,
    bindings: [
      {name: 'lonLat', buffer: lonLat, access: 'read'},
      {name: 'projected', buffer: projected, access: 'read_write'}
    ],
    body: `let lon = lonLat[lonLatOffset + index * 2u];
  let lat = lonLat[lonLatOffset + index * 2u + 1u];
  projected[projectedOffset + index * 2u] = ${world.width * 0.04} + (lon + 180.0) / 360.0 * ${world.width * 0.92};
  projected[projectedOffset + index * 2u + 1u] = 0.04 + (lat - ${SOUTH}.0) / ${NORTH - SOUTH}.0 * 0.92;`
  });
  return projected;
}
