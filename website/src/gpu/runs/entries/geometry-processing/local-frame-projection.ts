// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalFrameProjection turns a longitude/latitude graticule into meters in a local frame. The
// muted grid is the degrees laid out as-is (equirectangular); the bright grid is the projected
// meters the operation wrote, so the squeeze of the meridians with latitude is visible.

import {
  createGPULocalFrame,
  GPULocalFrameProjection
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addKernel} from './shared/shape-segments';
import {parameters} from './local-frame-projection.parameters';

const COLUMNS = 15;
const ROWS = 11;
const ORIGIN_LONGITUDE = 10;
const HALF_LONGITUDE = 36;
const HALF_LATITUDE = 26;
const WORLD_PER_DEGREE = 0.0145;
const METERS_PER_DEGREE = 111195;
const KINDS = ['azimuthal-equidistant', 'lambert-equal-area'] as const;

export default defineOperationRun({
  parameters,
  caption: 'A degree graticule projected to meters in a local frame.',
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
    const lonLat = builder.upload('lon-lat', 'float32x2', degrees);
    const meters = builder.allocate('meters', 'float32x2', COLUMNS * ROWS);
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
      const result = builder.allocate(id, 'float32x2', COLUMNS * ROWS);
      addKernel(builder, {
        id: `${id}-layout`,
        invocationCount: COLUMNS * ROWS,
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
        edges: builder.upload('edges', 'uint32x2', Uint32Array.from(edges))
      }
    };
  },
  rebuildOn: ['originLatitude', 'kind'],
  draw: ({outputs}) => [
    {layer: 'segments', context: true, nodes: outputs.degreeGrid, edges: outputs.edges},
    {layer: 'segments', nodes: outputs.meterGrid, edges: outputs.edges, radius: 2},
    {layer: 'dots', positions: outputs.meterGrid, radius: 3.5, color: [1, 0.6, 0.2, 1]}
  ]
});
