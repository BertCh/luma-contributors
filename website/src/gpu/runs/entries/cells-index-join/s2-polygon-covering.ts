// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUS2PolygonCovering covers a spherical polygon with a mixed-level S2 cell set: big cells deep
// inside, fine cells along the boundary. The white outline is the polygon.

import {createShoreRing} from '../../../data/hydrography';
import {GPUS2PolygonCovering} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCellFill} from './support/cells';
import {addS2Levels} from './support/kernels';
import {getUnitVector} from './support/polygons';
import {parameters} from './s2-polygon-covering.parameters';

const CAPACITY = 1536;

export default defineOperationRun({
  parameters,
  caption: 'Adaptive S2 cover: coarse cells inside, fine cells on the boundary.',
  createInputs: ({world}) => ({
    ring: createShoreRing(3, world.width * 0.5, 0.5, 0.45, 24),
    // The ring's own center, and where it sits now: the pointer drags it.
    origin: [world.width * 0.5, 0.5] as [number, number],
    center: [world.width * 0.5, 0.5] as [number, number]
  }),
  build(builder, inputs, values) {
    const vertexCount = inputs.ring.length;
    const positions = builder.allocate('positions', 'float32x3', vertexCount);
    const outline = builder.allocate('outline', 'float32x4', vertexCount);
    const ringOffsets = builder.upload('ring-offsets', 'uint32', Uint32Array.of(0, vertexCount));
    const cells = builder.allocate('cells', 'uint32x2', CAPACITY);
    const counts = builder.allocate('counts', 'uint32x2', 1);
    const status = builder.allocate('cover-status', 'uint32', 1);
    builder.graph.add(
      new GPUS2PolygonCovering({
        positions: positions.view,
        ringOffsets: ringOffsets.view,
        minimumLevel: 6,
        maximumLevel: values.maximumLevel,
        mode: values.mode === 0 ? 'exterior' : 'interior',
        capacity: CAPACITY,
        output: {cells: cells.view, counts: counts.view, status: status.view}
      })
    );
    return {
      outputs: {
        outline,
        counts,
        levels: addS2Levels(builder, 'cover', cells),
        fill: addCellFill(builder, 's2', cells)
      },
      update: current => {
        const ring = inputs.ring.map(([longitude, latitude]) => [
          inputs.center[0] + (longitude - inputs.origin[0]) * current.size,
          inputs.center[1] + (latitude - inputs.origin[1]) * current.size
        ]);
        positions.write(Float32Array.from(ring.flatMap(([x, y]) => getUnitVector(x, y))));
        outline.write(
          Float32Array.from(
            ring.flatMap(([x, y], index) => [x, y, ...ring[(index + 1) % vertexCount]])
          )
        );
      }
    };
  },
  rebuildOn: ['maximumLevel', 'mode'],
  // Drag: the polygon follows the pointer and the cover is recomputed on the GPU every frame.
  interaction: {
    kind: 'drag',
    hint: 'drag the polygon',
    rest: ({toUnit, inputs}) => toUnit(inputs.origin),
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.center = [pointer.x, pointer.y];
      const counts = peek(outputs.counts);
      const levels = peek(outputs.levels);
      let readout = 'cover …';
      if (counts && levels) {
        let minimum = 30;
        let maximum = 0;
        for (let row = 0; row < counts[0]; row++) {
          minimum = Math.min(minimum, levels[row]);
          maximum = Math.max(maximum, levels[row]);
        }
        readout = `cover ${counts[0]} cells · levels ${minimum} to ${maximum}`;
      }
      return {rerun: true, glyphs: [{shape: 'crosshair', at: inputs.center}], readout};
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.levels,
      triangleCount: outputs.fill.triangleCount,
      ramp: 'sequential',
      range: [5, 11],
      alpha: 0.75
    },
    {
      layer: 'triangles',
      positions: outputs.fill.boundaries,
      triangles: outputs.fill.wireframe,
      count: outputs.fill.wireframeCount,
      radius: 1,
      color: [1, 1, 1, 0.5]
    },
    {layer: 'lines', context: true, segments: outputs.outline, radius: 1.8, color: [1, 1, 1, 0.95]}
  ]
});
