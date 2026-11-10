// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalFrameProjection turns a longitude/latitude graticule into meters in a local frame. The
// muted grid is the degrees laid out as-is (equirectangular); the bright grid is the projected
// meters the operation wrote, so the squeeze of the meridians with latitude is visible.

import {
  createGPULocalFrame,
  GPULocalFrameProjection
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getCoastlineWindow} from '../../../data/coastline';
import {defineOperationRun} from '../../define-run';
import {addKernel} from './shared/shape-segments';
import {getBinCenter, getBinIndex} from './shared/pointer-support';
import {parameters} from './local-frame-projection.parameters';

const COLUMNS = 15;
const ROWS = 11;
const ORIGIN_LONGITUDE = 10;
const HALF_LONGITUDE = 36;
const HALF_LATITUDE = 26;
const WORLD_PER_DEGREE = 0.0145;
const METERS_PER_DEGREE = 111195;
const KINDS = ['azimuthal-equidistant', 'lambert-equal-area'] as const;
/** The frame origin is fixed at construction: the pointer settles on one of these latitudes. */
const ORIGIN_LATITUDES = [-40, -10, 20, 45, 60];

export default defineOperationRun({
  parameters,
  caption: 'Coastlines and a degree graticule projected into a local meter frame.',
  buildCache: 5,
  createInputs: ({world}) => ({world}),
  build(builder, {world}, values) {
    const originLatitude = values.originLatitude;
    const degrees = new Float32Array(COLUMNS * ROWS * 2);
    const edges: number[] = [];
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < COLUMNS; column++) {
        const node = row * COLUMNS + column;
        degrees[node * 2] = ORIGIN_LONGITUDE + ((column / (COLUMNS - 1)) * 2 - 1) * HALF_LONGITUDE;
        degrees[node * 2 + 1] = originLatitude + ((row / (ROWS - 1)) * 2 - 1) * HALF_LATITUDE;
        if (column + 1 < COLUMNS) edges.push(node, node + 1);
        if (row + 1 < ROWS) edges.push(node, node + COLUMNS);
      }
    }
    const coast = getCoastlineWindow(
      ORIGIN_LONGITUDE - HALF_LONGITUDE,
      originLatitude - HALF_LATITUDE,
      ORIGIN_LONGITUDE + HALF_LONGITUDE,
      originLatitude + HALF_LATITUDE
    );
    const gridCount = COLUMNS * ROWS;
    const vertexCount = gridCount + coast.length;
    const coastEdges = Uint32Array.from({length: coast.length}, (_, vertex) => gridCount + vertex);
    const lonLat = builder.upload(
      'lon-lat',
      'float32x2',
      Float32Array.from([...degrees, ...coast.flat()])
    );
    const meters = builder.allocate('meters', 'float32x2', vertexCount);
    builder.graph.add(
      new GPULocalFrameProjection({
        frame: createGPULocalFrame({
          origin: [ORIGIN_LONGITUDE, originLatitude],
          kind: KINDS[values.kind]
        }),
        positions: lonLat.view,
        output: meters.view,
        tolerance: 2000,
        bounds: [
          ORIGIN_LONGITUDE - HALF_LONGITUDE,
          originLatitude - HALF_LATITUDE,
          ORIGIN_LONGITUDE + HALF_LONGITUDE,
          originLatitude + HALF_LATITUDE
        ]
      })
    );
    const layout = (id: string, source: typeof lonLat, body: string) => {
      const result = builder.allocate(id, 'float32x2', vertexCount);
      addKernel(builder, {
        id: `${id}-layout`,
        invocationCount: vertexCount,
        bindings: [
          {name: 'source', buffer: source, access: 'read'},
          {name: 'result', buffer: result, access: 'read_write'}
        ],
        body
      });
      return result;
    };
    return {
      outputs: {
        degreeGrid: layout(
          'degree-grid',
          lonLat,
          `result[resultOffset + index * 2u] = ${world.width / 2} + (source[sourceOffset + index * 2u] - ${ORIGIN_LONGITUDE}.0) * ${WORLD_PER_DEGREE};
  result[resultOffset + index * 2u + 1u] = 0.5 + (source[sourceOffset + index * 2u + 1u] - ${originLatitude}.0) * ${WORLD_PER_DEGREE};`
        ),
        meterGrid: layout(
          'meter-grid',
          meters,
          `result[resultOffset + index * 2u] = ${world.width / 2} + source[sourceOffset + index * 2u] / ${METERS_PER_DEGREE}.0 * ${WORLD_PER_DEGREE};
  result[resultOffset + index * 2u + 1u] = 0.5 + source[sourceOffset + index * 2u + 1u] / ${METERS_PER_DEGREE}.0 * ${WORLD_PER_DEGREE};`
        ),
        coastEdges: builder.upload('coast-edges', 'uint32x2', coastEdges),
        edges: builder.upload('edges', 'uint32x2', Uint32Array.from(edges))
      }
    };
  },
  rebuildOn: ['originLatitude', 'kind'],
  // The pointer's height is the latitude of the frame origin (north is up); the grids re-center on it.
  interaction: {
    kind: 'place',
    hint: 'move the origin north or south',
    rest: [0.5, 1 - getBinCenter(ORIGIN_LATITUDES.indexOf(45), 1, ORIGIN_LATITUDES.length)],
    resolve(pointer, {values, world}) {
      const originLatitude =
        ORIGIN_LATITUDES[
          ORIGIN_LATITUDES.length - 1 - getBinIndex(pointer.unitY, 1, ORIGIN_LATITUDES.length)
        ];
      const origin: [number, number] = [world.width / 2, 0.5];
      return {
        settle: {originLatitude},
        glyphs: [
          {shape: 'dot', at: origin},
          {shape: 'line', from: origin, to: [world.width / 2, pointer.y], dashed: true},
          {shape: 'crosshair', at: [world.width / 2, pointer.y]}
        ],
        readout:
          originLatitude === values.originLatitude
            ? `origin ${originLatitude}°N · ${KINDS[values.kind]}`
            : `origin ${values.originLatitude}°N -> ${originLatitude}°N`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'segments',
      context: true,
      nodes: outputs.degreeGrid,
      edges: outputs.edges,
      alpha: 0.35
    },
    {
      layer: 'segments',
      context: true,
      nodes: outputs.degreeGrid,
      edges: outputs.coastEdges,
      radius: 2
    },
    {layer: 'segments', nodes: outputs.meterGrid, edges: outputs.edges, radius: 1, alpha: 0.6},
    {
      layer: 'segments',
      nodes: outputs.meterGrid,
      edges: outputs.coastEdges,
      radius: 2.5,
      color: [1, 0.6, 0.2, 1]
    }
  ]
});
