// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUDatumGridTransform,
  GPU_DATUM_GRID_TRANSFORM_STATUS
} from '@luma-contributors/analysis/geospatial-analysis';
import grid from '../../../data/midwest-nadcon.json';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {addKernelPass} from '../regression-weights/shared/kernel';
import type {RunGlyph} from '../../interaction';
import {parameters} from './datum-grid-transform.parameters';

const [west, south, east, north] = grid.bounds;
const COLUMNS = 12;
const ROWS = 6;
// The last location is the pointer's.
const POINT_COUNT = COLUMNS * ROWS + 1;

type Inputs = {samples: Float32Array; pointer: [number, number]};
type Outputs = {
  vectors: RunBuffer;
  tips: RunBuffer;
  meters: RunBuffer;
  boundaries: RunBuffer;
  origins: RunBuffer;
  exhausted: RunBuffer;
  pointerVector: RunBuffer;
};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  geographicWindow: [west, south, east, north],
  caption:
    'Midwest NADCON shifts. Lines join test locations to magnified shifts, colored by ground displacement; one location follows the pointer. Inverse direction: where the iteration limit stops it before the tolerance is met, the location is marked and has no vector.',
  createInputs: () => {
    const samples = new Float32Array(grid.width * grid.height * 3);
    for (let node = 0; node < grid.width * grid.height; node++) {
      samples[node * 3] = grid.eastNorthMilliarcseconds[node * 2] / 3600000;
      samples[node * 3 + 1] = grid.eastNorthMilliarcseconds[node * 2 + 1] / 3600000;
    }
    return {samples, pointer: [(west + east) / 2, (south + north) / 2] as [number, number]};
  },
  build(builder, inputs, values) {
    const positions = builder.allocate('test-locations', 'float32x3', POINT_COUNT);
    const locationWords = new Float32Array(POINT_COUNT * 3);
    // The low output retains the angular correction, even for sub-f32 displacements.
    const shift = builder.allocate('angular-shift', 'float32x3', POINT_COUNT);
    const magnification = builder.allocate('vector-magnification', 'float32', 1);
    const status = builder.allocate('status', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUDatumGridTransform({
        positions: positions.view,
        positionsLow: builder.upload('input-low', 'float32x3', new Float32Array(POINT_COUNT * 3))
          .view,
        samples: builder.upload('nadcon-degrees', 'float32x3', inputs.samples).view,
        mask: builder.upload(
          'validity',
          'uint32',
          new Uint32Array(grid.width * grid.height).fill(1)
        ).view,
        grids: [
          {
            width: grid.width,
            height: grid.height,
            sampleOffset: 0,
            origin: [west, south],
            spacing: [grid.spacing, grid.spacing]
          }
        ],
        operation: 'horizontal',
        direction: values.direction === 0 ? 'forward' : 'inverse',
        maximumIterations: values.maximumIterations,
        tolerance: 3e-9,
        output: {
          positions: builder.allocate('transformed-high', 'float32x3', POINT_COUNT).view,
          positionsLow: shift.view,
          status: status.view
        }
      })
    );
    const vectors = builder.allocate('display-vectors', 'float32x4', POINT_COUNT);
    const tips = builder.allocate('display-tips', 'float32x2', POINT_COUNT);
    const meters = builder.allocate('displacement-meters', 'float32', POINT_COUNT);
    const origins = builder.allocate('display-origins', 'float32x2', POINT_COUNT);
    const exhausted = builder.allocate('inverse-exhausted', 'float32', POINT_COUNT);
    addKernelPass(builder, {
      id: 'magnify-datum-vectors',
      invocationCount: POINT_COUNT,
      bindings: [
        {name: 'positions', buffer: positions, access: 'read'},
        {name: 'shift', buffer: shift, access: 'read'},
        {name: 'scale', buffer: magnification, access: 'read'},
        {name: 'vectors', buffer: vectors, access: 'read_write'},
        {name: 'tips', buffer: tips, access: 'read_write'},
        {name: 'meters', buffer: meters, access: 'read_write'}
      ],
      body: `let source = positionsOffset + index * 3u;
  let origin = vec2f(positions[source], positions[source + 1u]);
  let residual = shiftOffset + index * 3u;
  let delta = vec2f(shift[residual], shift[residual + 1u]);
  let tip = origin + delta * scale[scaleOffset];
  let line = vectorsOffset + index * 4u;
  vectors[line] = origin.x;
  vectors[line + 1u] = origin.y;
  vectors[line + 2u] = tip.x;
  vectors[line + 3u] = tip.y;
  tips[tipsOffset + index * 2u] = tip.x;
  tips[tipsOffset + index * 2u + 1u] = tip.y;
  meters[metersOffset + index] = 111320.0 * length(delta * vec2f(cos(radians(origin.y)), 1.0));`
    });
    addKernelPass(builder, {
      id: 'mark-exhausted-inverse',
      invocationCount: POINT_COUNT,
      bindings: [
        {name: 'positions', buffer: positions, access: 'read'},
        {name: 'status', buffer: status, access: 'read'},
        {name: 'origins', buffer: origins, access: 'read_write'},
        {name: 'exhausted', buffer: exhausted, access: 'read_write'}
      ],
      body: `let source = positionsOffset + index * 3u;
  origins[originsOffset + index * 2u] = positions[source];
  origins[originsOffset + index * 2u + 1u] = positions[source + 1u];
  exhausted[exhaustedOffset + index] = select(0.0, 1.0, (status[statusOffset + index] & ${GPU_DATUM_GRID_TRANSFORM_STATUS.inverseExhausted}u) != 0u);`
    });
    return {
      outputs: {
        pointerVector: builder.allocate('pointer-vector', 'float32x4', 1),
        vectors,
        tips,
        meters,
        origins,
        exhausted,
        boundaries: builder.upload(
          'state-boundaries',
          'float32x4',
          Float32Array.from(grid.stateBoundaries.flat())
        )
      },
      update: current => {
        for (let row = 0; row < ROWS; row++) {
          for (let column = 0; column < COLUMNS; column++) {
            const index = (row * COLUMNS + column) * 3;
            locationWords[index] = west + 0.7 + (column * (east - west - 1.4)) / (COLUMNS - 1);
            locationWords[index + 1] = south + 0.3 + (row * (north - south - 0.6)) / (ROWS - 1);
          }
        }
        locationWords.set([inputs.pointer[0], inputs.pointer[1], 0], (POINT_COUNT - 1) * 3);
        positions.write(locationWords);
        magnification.write(Float32Array.of(current.magnification));
      }
    };
  },
  rebuildOn: ['direction', 'maximumIterations'],
  // The pointer is one extra test location: the transform answers for it, and its magnified shift
  // vector is drawn as an arrow.
  interaction: {
    kind: 'place',
    hint: 'place a test location',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const vectors = peek(outputs.vectors);
      const last = (POINT_COUNT - 1) * 4;
      const glyphs: RunGlyph[] = [{shape: 'crosshair', at: [pointer.x, pointer.y], size: 8}];
      let shift = '';
      if (vectors) {
        const tip = [vectors[last + 2], vectors[last + 3]] as const;
        outputs.pointerVector.write(vectors.slice(last, last + 4));

        const east =
          ((tip[0] - vectors[last]) / values.magnification) *
          111320 *
          Math.cos((pointer.y * Math.PI) / 180);
        const north = ((tip[1] - vectors[last + 1]) / values.magnification) * 111320;
        // A callout arrow in the shift's direction, long enough to read at any shift (1 degree plus 6 degrees per 100 m).
        const length = Math.hypot(east, north) || 1;
        const reach = 1 + (length / 100) * 6;
        glyphs.push({shape: 'ring', at: [pointer.x, pointer.y], radius: 0.45, dashed: true});
        glyphs.push({
          shape: 'arrow',
          from: [pointer.x, pointer.y],
          to: [pointer.x + (east / length) * reach, pointer.y + (north / length) * reach]
        });
        shift = ` · ${Math.abs(east).toFixed(1)} m ${east >= 0 ? 'E' : 'W'}, ${Math.abs(north).toFixed(1)} m ${north >= 0 ? 'N' : 'S'}`;
      }
      return {
        rerun: true,
        glyphs,
        readout: `${Math.abs(pointer.x).toFixed(1)}°${pointer.x < 0 ? 'W' : 'E'} ${pointer.y.toFixed(1)}°N${shift}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.boundaries,
      color: [...theme.muted, 0.6],
      radius: 1
    },
    {layer: 'lines', segments: outputs.vectors, color: [...theme.ink, 0.8], radius: 1.7},
    {
      layer: 'lines',
      segments: outputs.pointerVector,
      count: 1,
      color: [1, 0.75, 0.2, 1],
      radius: 3.4
    },
    {layer: 'dots', positions: outputs.tips, values: outputs.meters, radius: 6, range: [0, 30]},
    // Locations where the inverse iteration stopped before reaching its tolerance: no shift is written.
    {
      layer: 'dots',
      positions: outputs.origins,
      values: outputs.exhausted,
      range: [0, 1],
      fade: true,
      radius: 9,
      ramp: 'diverging'
    }
  ]
});
